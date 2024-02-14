import dataclasses
import json
import os
import sys
import tarfile
import typing
import warnings
from collections import defaultdict
from contextlib import suppress
from functools import partial
from io import BufferedIOBase, BytesIO, IOBase, RawIOBase, TextIOBase
from pathlib import Path

import numpy as np
import tifffile
from local_migrator import update_argument
from pydantic import Field

from PartSegCore.algorithm_describe_base import AlgorithmProperty, Register, ROIExtractionProfile
from PartSegCore.io_utils import (
    IO_MASK_METADATA_FILE,
    LoadBase,
    LoadPoints,
    SaveBase,
    SaveMaskAsTiff,
    SaveROIAsNumpy,
    SaveROIAsTIFF,
    SegmentationType,
    WrongFileTypeException,
    check_segmentation_type,
    get_tarinfo,
    load_metadata_base,
    open_tar_file,
    proxy_callback,
    tar_to_buff,
)
from PartSegCore.json_hooks import PartSegEncoder
from PartSegCore.project_info import AdditionalLayerDescription, HistoryElement
from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import BaseModel
from PartSegImage import BaseImageWriter, GenericImageReader, Image, IMAGEJImageWriter, ImageWriter, TiffImageReader
from PartSegImage.image import FRAME_THICKNESS, reduce_array

try:
    from napari_builtins.io import napari_write_points
except ImportError:
    from napari.plugins._builtins import napari_write_points

if sys.version_info[:3] != (3, 9, 7):
    from PartSegCore.project_info import ProjectInfoBase
else:  # pragma: no cover
    ProjectInfoBase = object


def empty_fun(_a0=None, _a1=None):
    """
    This is empty fun to pass as callback to for report.
    """


@dataclasses.dataclass(frozen=True)
class MaskProjectTuple(ProjectInfoBase):
    """
    Dataclass instance to describe segmentation state

    :ivar str file_path: path to current processed file
    :ivar typing.Union[Image,str,None] ~.image: image which is proceeded in given segmentation.
        If :py:class:`str` then it is path to image on drive
    :ivar typing.Optional[np.ndarray] ~.mask: Mask limiting segmentation area.
    :ivar ROIInfo ~.roi_info: ROI information.
    :ivar SegmentationInfo ~.roi_info: ROI description
    :ivar typing.List[int] ~.selected_components: list of selected components
    :ivar typing.Dict[int,typing.Optional[SegmentationProfile]] ~.segmentation_parameters:
        For each component description set of parameters used for segmentation
    :ivar typing.List[HistoryElement] history: list of operations needed to create :py:attr:`mask`
    :ivar str ~.errors: information about problems meet during calculation
    :ivar typing.Optional[typing.List[float]] ~.spacing: information about spacing when image is missed.
        For napari read plugin
    """

    file_path: str
    image: typing.Union[Image, str, None]
    mask: typing.Optional[np.ndarray] = None
    roi_info: ROIInfo = dataclasses.field(default_factory=lambda: ROIInfo(None))
    additional_layers: typing.Dict[str, AdditionalLayerDescription] = dataclasses.field(default_factory=dict)
    selected_components: typing.List[int] = dataclasses.field(default_factory=list)
    roi_extraction_parameters: typing.Dict[int, typing.Optional[ROIExtractionProfile]] = dataclasses.field(
        default_factory=dict
    )
    history: typing.List[HistoryElement] = dataclasses.field(default_factory=list)
    errors: str = ""
    spacing: typing.Optional[typing.List[float]] = None
    points: typing.Optional[np.ndarray] = None
    frame_thickness: int = FRAME_THICKNESS

    def get_raw_copy(self):
        return MaskProjectTuple(self.file_path, self.image.substitute(mask=None))

    def is_raw(self):
        return self.roi_info.roi is None

    def is_masked(self):
        return self.mask is not None

    def get_raw_mask_copy(self):
        return MaskProjectTuple(file_path=self.file_path, image=self.image.substitute(), mask=self.mask)

    @property
    def roi(self):  # pragma: no cover
        warnings.warn("roi is deprecated", DeprecationWarning, stacklevel=2)
        return self.roi_info.roi


class SaveROIOptions(BaseModel):
    relative_path: bool = Field(
        True, title="Relative Path\nin segmentation", description="Use relative path to image in segmentation file"
    )
    mask_data: bool = Field(
        False,
        title="Keep data outside ROI",
        description="When loading data in ROI analysis, if not checked"
        " then data outside ROI will be replaced with zeros.",
    )
    frame_thickness: int = Field(2, title="Frame thickness", description="Thickness of frame around ROI")
    spacing: typing.List[float] = Field([10**-6, 10**-6, 10**-6], hidden=True)


def _save_mask_roi(project: MaskProjectTuple, tar_file: tarfile.TarFile, parameters: SaveROIOptions):
    segmentation_buff = BytesIO()
    # noinspection PyTypeChecker
    if project.image is not None:
        spacing = project.image.spacing
    else:
        spacing = parameters.spacing
    segmentation_image = Image(project.roi_info.roi, spacing, axes_order=Image.axis_order.replace("C", ""))
    try:
        ImageWriter.save(segmentation_image, segmentation_buff, compression=None)
    except ValueError:
        segmentation_buff.seek(0)
        tifffile.imwrite(segmentation_buff, project.roi_info.roi)
    segmentation_tar = get_tarinfo("segmentation.tif", segmentation_buff)
    tar_file.addfile(segmentation_tar, fileobj=segmentation_buff)


def _save_mask_roi_metadata(
    project: MaskProjectTuple, tar_file: tarfile.TarFile, parameters: SaveROIOptions, file_data
):
    metadata = {
        "components": [int(x) for x in project.selected_components],
        "parameters": {str(k): v for k, v in project.roi_extraction_parameters.items()},
        "shape": project.roi_info.roi.shape,
        "annotations": project.roi_info.annotations,
        "keep_data_outside_mask": not parameters.mask_data,
        "frame_thickness": parameters.frame_thickness,
    }
    if isinstance(project.image, Image):
        file_path = project.image.file_path
    elif isinstance(project.image, str):
        file_path = project.image
    else:
        file_path = ""
    if file_path:
        if parameters.relative_path and isinstance(file_data, str):
            metadata["base_file"] = os.path.relpath(file_path, os.path.dirname(file_data))
        else:
            metadata["base_file"] = file_path
    metadata_buff = BytesIO(json.dumps(metadata, cls=PartSegEncoder).encode("utf-8"))
    metadata_tar = get_tarinfo(IO_MASK_METADATA_FILE, metadata_buff)
    tar_file.addfile(metadata_tar, metadata_buff)


def _save_mask_mask(project: MaskProjectTuple, tar_file: tarfile.TarFile):
    mask = project.mask
    if mask.dtype == bool:
        mask = mask.astype(np.uint8)
    mask_buff = BytesIO()
    tifffile.imwrite(mask_buff, mask)
    mask_tar = get_tarinfo("mask.tif", mask_buff)
    tar_file.addfile(mask_tar, fileobj=mask_buff)


def _save_mask_alternative(project: MaskProjectTuple, tar_file: tarfile.TarFile):
    alternative_buff = BytesIO()
    np.savez(alternative_buff, **project.roi_info.alternative)
    alternative_tar = get_tarinfo("alternative.npz", alternative_buff)
    tar_file.addfile(alternative_tar, fileobj=alternative_buff)


def _save_mask_history(project: MaskProjectTuple, tar_file: tarfile.TarFile):
    el_info = []
    for i, hist in enumerate(project.history):
        el_info.append(
            {
                "index": i,
                "mask_property": hist.mask_property,
                "segmentation_parameters": hist.roi_extraction_parameters,
                "annotations": hist.annotations,
            }
        )
        hist.arrays.seek(0)
        hist_info = get_tarinfo(f"history/arrays_{i}.npz", hist.arrays)
        hist.arrays.seek(0)
        tar_file.addfile(hist_info, hist.arrays)
    if el_info:
        hist_str = json.dumps(el_info, cls=PartSegEncoder)
        hist_buff = BytesIO(hist_str.encode("utf-8"))
        tar_algorithm = get_tarinfo("history/history.json", hist_buff)
        tar_file.addfile(tar_algorithm, hist_buff)


def save_stack_segmentation(
    file_data: typing.Union[tarfile.TarFile, str, Path, TextIOBase, BufferedIOBase, RawIOBase, IOBase],
    segmentation_info: MaskProjectTuple,
    parameters: SaveROIOptions,
    range_changed=empty_fun,
    step_changed=empty_fun,
):
    range_changed(0, 7)
    tar_file, _file_path = open_tar_file(file_data, "w:gz")
    step_changed(1)
    try:
        _save_mask_roi(segmentation_info, tar_file, parameters)
        step_changed(2)
        _save_mask_roi_metadata(segmentation_info, tar_file, parameters, file_data)
        step_changed(3)
        if segmentation_info.mask is not None:
            _save_mask_mask(segmentation_info, tar_file)
        if segmentation_info.roi_info.alternative:
            _save_mask_alternative(segmentation_info, tar_file)
        step_changed(4)
        _save_mask_history(segmentation_info, tar_file)
        step_changed(5)

    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()
    step_changed(6)


def load_stack_segmentation_from_tar(tar_file: tarfile.TarFile, file_path: str, step_changed=None):
    if check_segmentation_type(tar_file) != SegmentationType.mask:
        raise WrongFileTypeException  # pragma: no cover
    files = tar_file.getnames()
    step_changed(1)
    metadata = load_metadata(tar_file.extractfile(IO_MASK_METADATA_FILE).read().decode("utf8"))
    step_changed(2)
    if "segmentation.npy" in files:
        segmentation_file_name = "segmentation.npy"
        segmentation_load_fun = np.load
    else:
        segmentation_file_name = "segmentation.tif"
        segmentation_load_fun = TiffImageReader.read_image
    segmentation_buff = BytesIO()
    segmentation_tar = tar_file.extractfile(tar_file.getmember(segmentation_file_name))
    segmentation_buff.write(segmentation_tar.read())
    step_changed(3)
    segmentation_buff.seek(0)
    roi = segmentation_load_fun(segmentation_buff)
    if isinstance(roi, Image):
        spacing = roi.spacing
        roi = roi.get_channel(0)
    else:
        spacing = None
    step_changed(4)
    if "mask.tif" in tar_file.getnames():
        mask = tifffile.imread(tar_to_buff(tar_file, "mask.tif"))
        if np.max(mask) == 1:
            mask = mask.astype(bool)
    else:
        mask = None
    if "alternative.npz" in tar_file.getnames():
        alternative = np.load(tar_to_buff(tar_file, "alternative.npz"))
    else:
        alternative = {}
    roi_info = ROIInfo(reduce_array(roi), annotations=metadata.get("annotations", {}), alternative=alternative)
    step_changed(5)
    history = []
    with suppress(KeyError):
        history_buff = tar_file.extractfile(tar_file.getmember("history/history.json")).read()
        history_json = load_metadata(history_buff)
        for el in history_json:
            history_buffer = BytesIO()
            history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
            history_buffer.seek(0)
            history.append(
                HistoryElement(
                    roi_extraction_parameters=el["segmentation_parameters"],
                    mask_property=el["mask_property"],
                    arrays=history_buffer,
                    annotations=el.get("annotations", {}),
                )
            )
    step_changed(6)
    return MaskProjectTuple(
        file_path=file_path,
        image=metadata.get("base_file"),
        roi_info=roi_info,
        selected_components=metadata["components"],
        mask=mask,
        roi_extraction_parameters=metadata.get("parameters"),
        history=history,
        spacing=([10 ** (-9), *list(spacing)]) if spacing is not None else None,
        frame_thickness=metadata.get("frame_thickness", FRAME_THICKNESS),
    )


def load_stack_segmentation(file_data: typing.Union[str, Path], range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 7)
    tar_file = open_tar_file(file_data)[0]
    try:
        return load_stack_segmentation_from_tar(
            tar_file, file_data if isinstance(file_data, str) else "", step_changed=step_changed
        )
    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()


class LoadROI(LoadBase):
    """
    Load ROI segmentation data.
    """

    @classmethod
    def get_name(cls):
        return "Segmentation to image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        segmentation_tuple = load_stack_segmentation(
            load_locations[0], range_changed=range_changed, step_changed=step_changed
        )
        if segmentation_tuple.roi_extraction_parameters is None:
            parameters = defaultdict(lambda: None)
        else:
            parameters = defaultdict(
                lambda: None,
                [(int(k), v) for k, v in segmentation_tuple.roi_extraction_parameters.items()],
            )
        return dataclasses.replace(segmentation_tuple, roi_extraction_parameters=parameters)

    @classmethod
    def partial(cls):
        return False


class LoadROIParameters(LoadBase):
    """
    Load parameters of ROI segmentation. From segmentation file or from json
    """

    @classmethod
    def get_name(cls):
        return "Segmentation parameters (*.json *.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg_par"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        file_data = load_locations[0]

        if isinstance(file_data, (str, Path)):
            ext = os.path.splitext(file_data)[1]
            if ext == ".json":
                project_metadata = load_metadata(file_data)
                if isinstance(project_metadata, ROIExtractionProfile):
                    parameters = {1: project_metadata}
                else:
                    parameters = defaultdict(
                        lambda: None,
                        [(int(k), v) for k, v in project_metadata["parameters"].items()],
                    )
                return MaskProjectTuple(file_path=file_data, image=None, roi_extraction_parameters=parameters)

        tar_file, _ = open_tar_file(file_data)
        try:
            project_metadata = load_metadata(tar_file.extractfile(IO_MASK_METADATA_FILE).read().decode("utf8"))
            parameters = defaultdict(
                lambda: None,
                [(int(k), v) for k, v in project_metadata["parameters"].items()],
            )
        finally:
            if isinstance(file_data, (str, Path)):
                tar_file.close()
        return MaskProjectTuple(file_path=file_data, image=None, roi_extraction_parameters=parameters)


class LoadROIImage(LoadBase):
    """
    Load ROI segmentation and image which is pointed in.
    """

    @classmethod
    def get_name(cls):
        return "ROI project with image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "roi_image"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        seg = LoadROI.load(load_locations)
        base_file = load_locations[1] if len(load_locations) > 1 else seg.image
        if base_file is None:
            raise OSError("base file for segmentation not defined")
        if os.path.isabs(base_file):
            file_path = base_file
        elif not isinstance(load_locations[0], str):
            raise OSError(f"Cannot use relative path {base_file} for non path argument")
        else:
            file_path = os.path.join(os.path.dirname(load_locations[0]), base_file)
        if not os.path.exists(file_path):
            raise OSError(f"Base file for segmentation do not exists: {base_file} -> {file_path}")
        if metadata is None:
            metadata = {"default_spacing": (10**-6, 10**-6, 10**-6)}
        image = GenericImageReader.read_image(
            file_path,
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )
        return dataclasses.replace(
            seg, file_path=image.file_path, image=image, roi_info=seg.roi_info.fit_to_image(image)
        )


class LoadStackImage(LoadBase):
    """
    Load image from standard microscopy images
    """

    @classmethod
    def get_name(cls):
        return "Image(*.tif *.tiff *.lsm *.czi *.oib *.oif *.obsep)"

    @classmethod
    def get_short_name(cls):
        return "img"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        if metadata is None:
            metadata = {"default_spacing": (10**-6, 10**-6, 10**-6)}
        image = GenericImageReader.read_image(
            load_locations[0],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )

        return MaskProjectTuple(image.file_path, image, selected_components=[])


class LoadStackImageWithMask(LoadBase):
    """
    Load image, then mask from secondary file
    """

    @classmethod
    def get_short_name(cls):
        return "img_with_mask"

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        base, ext = os.path.splitext(file_paths[0])
        return f"{base}_mask{ext}"

    @classmethod
    def number_of_files(cls):
        return 2

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectInfoBase, typing.List[ProjectInfoBase]]:
        if metadata is None:
            metadata = {"default_spacing": (10**-6, 10**-6, 10**-6)}
        image = GenericImageReader.read_image(
            load_locations[0],
            load_locations[1],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )

        return MaskProjectTuple(image.file_path, image, mask=image.mask, selected_components=[])

    @classmethod
    def get_name(cls) -> str:
        return "Image with mask(*.tif *.tiff *.lsm *.czi *.oib *.oif)"


class SaveROI(SaveBase):
    """
    Save current ROI as a project
    """

    __argument_class__ = SaveROIOptions

    @classmethod
    def get_name(cls):
        return "ROI project (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    @update_argument("parameters")
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: MaskProjectTuple,
        parameters: SaveROIOptions,
        range_changed=None,
        step_changed=None,
    ):
        save_stack_segmentation(save_location, project_info, parameters)


class SaveComponentsOptions(BaseModel):
    frame: int = Field(0, title="Frame", description="How many pixels around bounding box of ROI should be saved")
    mask_data: bool = Field(
        False,
        title="Keep data outside ROI",
        description="If not checked then data outside ROI will be replaced with zeros.",
    )


def save_components(
    image: Image,
    components: list,
    dir_path: str,
    roi_info: ROIInfo,
    parameters: typing.Optional[SaveComponentsOptions] = None,
    points: typing.Optional[np.ndarray] = None,
    range_changed=None,
    step_changed=None,
    writer_class: typing.Type[BaseImageWriter] = ImageWriter,
):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun

    if parameters is None:
        parameters = SaveComponentsOptions()

    roi_info = roi_info.fit_to_image(image)
    os.makedirs(dir_path, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(image.file_path))[0]
    important_axis = "XY" if image.is_2d else "XYZ"
    index_to_frame_points = image.calc_index_to_frame(image.axis_order, important_axis)
    points_casted = points.astype(np.uint16) if points is not None else None
    if not components:
        components = list(roi_info.bound_info.keys())
    range_changed(0, 2 * len(components))
    for i in components:
        components_mark = np.array(roi_info.roi == i)
        im = image.cut_image(
            components_mark, replace_mask=True, frame=parameters.frame, zero_out_cut_area=parameters.mask_data
        )
        if points is not None and points_casted is not None:
            points_mask = components_mark[tuple(points_casted.T)]
            filtered_points = points[points_mask]
            filtered_points[:, 1] = np.round(filtered_points[:, 1])
            lower_bound = np.min(np.nonzero(components_mark), axis=1)
            for j in index_to_frame_points:
                lower_bound[j] -= FRAME_THICKNESS
            napari_write_points(
                os.path.join(dir_path, f"{file_name}_component{i}.csv"), filtered_points - lower_bound, {}
            )

        writer_class.save(im, os.path.join(dir_path, f"{file_name}_component{i}.tif"))
        step_changed(2 * i + 1)
        writer_class.save_mask(im, os.path.join(dir_path, f"{file_name}_component{i}_mask.tif"))
        step_changed(2 * i + 2)


class SaveComponents(SaveBase):
    """
    Save selected components in separated files.
    """

    __argument_class__ = SaveComponentsOptions

    @classmethod
    def get_short_name(cls):
        return "comp"

    @classmethod
    @update_argument("parameters")
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: MaskProjectTuple,
        parameters: SaveComponentsOptions,
        range_changed=None,
        step_changed=None,
    ):
        save_components(
            project_info.image,
            project_info.selected_components,
            save_location,
            project_info.roi_info,
            parameters,
            project_info.points,
            range_changed,
            step_changed,
        )

    @classmethod
    def get_name(cls) -> str:
        return "Components"


class SaveComponentsImagej(SaveBase):
    __argument_class__ = SaveComponentsOptions

    @classmethod
    def get_short_name(cls):
        return "comp_imagej"

    @classmethod
    @update_argument("parameters")
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: MaskProjectTuple,
        parameters: SaveComponentsOptions,
        range_changed=None,
        step_changed=None,
    ):
        save_components(
            project_info.image,
            project_info.selected_components,
            save_location,
            project_info.roi_info,
            parameters,
            project_info.points,
            range_changed,
            step_changed,
            writer_class=IMAGEJImageWriter,
        )

    @classmethod
    def get_name(cls) -> str:
        return "Components Imagej Tiff"


class SaveParametersJSON(SaveBase):
    """
    Save parameters of roi mask segmentation
    """

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: typing.Union[ROIExtractionProfile, MaskProjectTuple],
        parameters: typing.Optional[dict] = None,
        range_changed=None,
        step_changed=None,
    ):
        """
        :param save_location: path to save
        :param project_info: data to save in json file
        :param parameters: Not used, keep for satisfy interface
        :param range_changed: Not used, keep for satisfy interface
        :param step_changed: Not used, keep for satisfy interface
        :return:
        """
        with open(save_location, "w", encoding="utf-8") as ff:
            if isinstance(project_info, ROIExtractionProfile):
                json.dump(project_info, ff, cls=PartSegEncoder)
            else:
                json.dump({"parameters": project_info.roi_extraction_parameters}, ff, cls=PartSegEncoder)

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []

    @classmethod
    def get_short_name(cls):
        return "json"

    @classmethod
    def get_name(cls) -> str:
        return "Parameters (*.json)"


class LoadROIFromTIFF(LoadBase):
    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectInfoBase, typing.List[ProjectInfoBase]]:
        image = TiffImageReader.read_image(load_locations[0])
        roi = image.get_channel(0)
        return MaskProjectTuple(
            file_path=load_locations[0],
            image=None,
            roi_info=ROIInfo(roi),
            roi_extraction_parameters=defaultdict(lambda: None),
        )

    @classmethod
    def get_short_name(cls):
        return "roi_tiff"

    @classmethod
    def get_name(cls) -> str:
        return "ROI from tiff (*.tif *.tiff)"


def load_metadata(data: typing.Union[str, Path, typing.TextIO]):
    """
    Load metadata saved in json format for segmentation mask
    :param data: path to json file, string with json, or opened file
    :return: restored structures
    """
    return load_metadata_base(data)


load_dict = Register(
    LoadStackImage, LoadROIImage, LoadStackImageWithMask, LoadPoints, class_methods=LoadBase.need_functions
)
save_parameters_dict = Register(SaveParametersJSON, class_methods=SaveBase.need_functions)
save_components_dict = Register(SaveComponents, SaveComponentsImagej, class_methods=SaveBase.need_functions)
save_segmentation_dict = Register(
    SaveROI, SaveMaskAsTiff, SaveROIAsTIFF, SaveROIAsNumpy, class_methods=SaveBase.need_functions
)
