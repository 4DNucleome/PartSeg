import dataclasses
import json
import os
import tarfile
import typing
from collections import defaultdict
from functools import partial
from io import BufferedIOBase, BytesIO, IOBase, RawIOBase, TextIOBase
from pathlib import Path

import numpy as np
import tifffile
from napari.plugins._builtins import napari_write_points

from PartSegImage import GenericImageReader, Image, ImageWriter, TiffImageReader
from PartSegImage.image import FRAME_THICKNESS, reduce_array

from ..algorithm_describe_base import AlgorithmProperty, Register, ROIExtractionProfile
from ..io_utils import (
    LoadBase,
    LoadPoints,
    SaveBase,
    SaveMaskAsTiff,
    SaveROIAsNumpy,
    SaveROIAsTIFF,
    SegmentationType,
    UpdateLoadedMetadataBase,
    WrongFileTypeException,
    check_segmentation_type,
    get_tarinfo,
    open_tar_file,
    proxy_callback,
    tar_to_buff,
)
from ..json_hooks import ProfileEncoder
from ..project_info import AdditionalLayerDescription, HistoryElement, ProjectInfoBase
from ..roi_info import ROIInfo


@dataclasses.dataclass(frozen=True)
class MaskProjectTuple(ProjectInfoBase):
    """
    Dataclass instance to describe segmentation state

    :ivar str file_path: path to current processed file
    :ivar typing.Union[Image,str,None] ~.image: image which is proceeded in given segmentation.
        If :py:class:`str` then it is path to image on drive
    :ivar typing.Optional[np.ndarray] ~.mask: Mask limiting segmentation area.
    :ivar typing.Optional[np.ndarray] ~.roi: ROI array.
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
    roi_info: ROIInfo = ROIInfo(None)
    additional_layers: typing.Dict[str, AdditionalLayerDescription] = dataclasses.field(default_factory=dict)
    selected_components: typing.List[int] = dataclasses.field(default_factory=list)
    roi_extraction_parameters: typing.Dict[int, typing.Optional[ROIExtractionProfile]] = dataclasses.field(
        default_factory=dict
    )
    history: typing.List[HistoryElement] = dataclasses.field(default_factory=list)
    errors: str = ""
    spacing: typing.Optional[typing.List[float]] = None
    points: typing.Optional[np.ndarray] = None

    def get_raw_copy(self):
        return MaskProjectTuple(self.file_path, self.image.substitute(mask=None))

    def is_raw(self):
        return self.roi is None

    def is_masked(self):
        return self.mask is not None

    def get_raw_mask_copy(self):
        return MaskProjectTuple(file_path=self.file_path, image=self.image.substitute(), mask=self.mask)


def save_stack_segmentation(
    file_data: typing.Union[tarfile.TarFile, str, Path, TextIOBase, BufferedIOBase, RawIOBase, IOBase],
    segmentation_info: MaskProjectTuple,
    parameters: dict,
    range_changed=None,
    step_changed=None,
):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 7)
    tar_file, file_path = open_tar_file(file_data, "w")
    step_changed(1)
    try:
        segmentation_buff = BytesIO()
        # noinspection PyTypeChecker
        if segmentation_info.image is not None:
            spacing = segmentation_info.image.spacing
        else:
            spacing = parameters.get("spacing", (10 ** -6, 10 ** -6, 10 ** -6))
        segmentation_image = Image(
            segmentation_info.roi_info.roi, spacing, axes_order=Image.axis_order.replace("C", "")
        )
        try:
            ImageWriter.save(segmentation_image, segmentation_buff)
        except ValueError:
            segmentation_buff.seek(0)
            tifffile.imwrite(segmentation_buff, segmentation_info.roi_info.roi, compress=9)
        segmentation_tar = get_tarinfo("segmentation.tif", segmentation_buff)
        tar_file.addfile(segmentation_tar, fileobj=segmentation_buff)
        step_changed(3)
        metadata = {
            "components": [int(x) for x in segmentation_info.selected_components],
            "parameters": {str(k): v for k, v in segmentation_info.roi_extraction_parameters.items()},
            "shape": segmentation_info.roi_info.roi.shape,
            "annotations": segmentation_info.roi_info.annotations,
        }
        if isinstance(segmentation_info.image, Image):
            file_path = segmentation_info.image.file_path
        elif isinstance(segmentation_info.image, str):
            file_path = segmentation_info.image
        else:
            file_path = ""
        if file_path != "":
            if parameters["relative_path"] and isinstance(file_data, str):
                metadata["base_file"] = os.path.relpath(file_path, os.path.dirname(file_data))
            else:
                metadata["base_file"] = file_path
        metadata_buff = BytesIO(json.dumps(metadata, cls=ProfileEncoder).encode("utf-8"))
        metadata_tar = get_tarinfo("metadata.json", metadata_buff)
        tar_file.addfile(metadata_tar, metadata_buff)
        step_changed(4)
        if segmentation_info.mask is not None:
            mask = segmentation_info.mask
            if mask.dtype == bool:
                mask = mask.astype(np.uint8)
            mask_buff = BytesIO()
            tifffile.imwrite(mask_buff, mask, compress=9)
            mask_tar = get_tarinfo("mask.tif", mask_buff)
            tar_file.addfile(mask_tar, fileobj=mask_buff)
        if segmentation_info.roi_info.alternative:
            alternative_buff = BytesIO()
            np.savez(alternative_buff, **segmentation_info.roi_info.alternative)
            alternative_tar = get_tarinfo("alternative.npz", alternative_buff)
            tar_file.addfile(alternative_tar, fileobj=alternative_buff)
        step_changed(5)
        el_info = []
        for i, hist in enumerate(segmentation_info.history):
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
            hist_str = json.dumps(el_info, cls=ProfileEncoder)
            hist_buff = BytesIO(hist_str.encode("utf-8"))
            tar_algorithm = get_tarinfo("history/history.json", hist_buff)
            tar_file.addfile(tar_algorithm, hist_buff)
        step_changed(6)

    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()
    step_changed(6)


def load_stack_segmentation(file_data: typing.Union[str, Path], range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 7)
    tar_file = open_tar_file(file_data)[0]
    try:
        if check_segmentation_type(tar_file) != SegmentationType.mask:
            raise WrongFileTypeException()
        files = tar_file.getnames()
        step_changed(1)
        metadata = load_metadata(tar_file.extractfile("metadata.json").read().decode("utf8"))
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
        try:
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

        except KeyError:
            pass
        step_changed(6)
    finally:
        if isinstance(file_data, (str, Path)):
            tar_file.close()
    return MaskProjectTuple(
        file_path=file_data if isinstance(file_data, str) else "",
        image=metadata["base_file"] if "base_file" in metadata else None,
        roi_info=roi_info,
        selected_components=metadata["components"],
        mask=mask,
        roi_extraction_parameters=metadata["parameters"] if "parameters" in metadata else None,
        history=history,
        spacing=([10 ** -9] + list(spacing)) if spacing is not None else None,
    )


def empty_fun(_a0=None, _a1=None):
    """
    This is empty fun to pass as callback to for report.
    """


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

    @staticmethod
    def fix_parameters(profile: ROIExtractionProfile):
        if profile is None:
            return
        if (profile.algorithm in {"Threshold", "Auto Threshold"}) and isinstance(profile.values["smooth_border"], bool):
            if profile.values["smooth_border"] and "smooth_border_radius" in profile.values:
                profile.values["smooth_border"] = {
                    "name": "Opening",
                    "values": {"smooth_border_radius": profile.values["smooth_border_radius"]},
                }
                del profile.values["smooth_border_radius"]
            else:
                profile.values["smooth_border"] = {"name": "None", "values": {}}
        return profile

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
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
                [(int(k), cls.fix_parameters(v)) for k, v in segmentation_tuple.roi_extraction_parameters.items()],
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
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        file_data = load_locations[0]

        if isinstance(file_data, (str, Path)):
            ext = os.path.splitext(file_data)[1]
            if ext == ".json":
                project_metadata = load_metadata(file_data)
                if isinstance(metadata, ROIExtractionProfile):
                    parameters = {1: metadata}
                else:
                    parameters = defaultdict(
                        lambda: None,
                        [(int(k), LoadROI.fix_parameters(v)) for k, v in project_metadata["parameters"].items()],
                    )
                return MaskProjectTuple(file_path=file_data, image=None, roi_extraction_parameters=parameters)

        tar_file, _ = open_tar_file(file_data)
        try:
            project_metadata = load_metadata(tar_file.extractfile("metadata.json").read().decode("utf8"))
            parameters = defaultdict(
                lambda: None,
                [(int(k), LoadROI.fix_parameters(v)) for k, v in project_metadata["parameters"].items()],
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
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        seg = LoadROI.load(load_locations)
        base_file = load_locations[1] if len(load_locations) > 1 else seg.image
        if base_file is None:
            raise OSError("base file for segmentation not defined")
        if os.path.isabs(base_file):
            file_path = base_file
        else:
            if not isinstance(load_locations[0], str):
                raise OSError(f"Cannot use relative path {base_file} for non path argument")
            file_path = os.path.join(os.path.dirname(load_locations[0]), base_file)
        if not os.path.exists(file_path):
            raise OSError(f"Base file for segmentation do not exists: {base_file} -> {file_path}")
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        image = GenericImageReader.read_image(
            file_path,
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )
        # noinspection PyProtectedMember
        # image.file_path = load_locations[0]
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
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> MaskProjectTuple:
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        image = GenericImageReader.read_image(
            load_locations[0],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"],
        )

        return MaskProjectTuple(image.file_path, image, selected_components=[])


class LoadStackImageWithMask(LoadBase):
    """
    Load image, hne mask from secondary file
    """

    @classmethod
    def get_short_name(cls):
        return "img_with_mask"

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        base, ext = os.path.splitext(file_paths[0])
        return base + "_mask" + ext

    @classmethod
    def number_of_files(cls):
        return 2

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectInfoBase, typing.List[ProjectInfoBase]]:
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
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
    Save current ROI
    """

    @classmethod
    def get_name(cls):
        return "ROI project (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("relative_path", "Relative Path\nin segmentation", False)]

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: MaskProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        save_stack_segmentation(save_location, project_info, parameters)


def save_components(
    image: Image,
    components: list,
    dir_path: str,
    roi_info: ROIInfo,
    points: typing.Optional[np.ndarray] = None,
    range_changed=None,
    step_changed=None,
):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun

    roi_info = roi_info.fit_to_image(image)
    os.makedirs(dir_path, exist_ok=True)

    file_name = os.path.splitext(os.path.basename(image.file_path))[0]
    points_casted = None
    important_axis = "XY" if image.is_2d else "XYZ"
    index_to_frame_points = image.calc_index_to_frame(image.axis_order, important_axis)
    if points is not None:
        points_casted = points.astype(np.uint16)

    range_changed(0, 2 * len(components))
    for i in components:
        components_mark = np.array(roi_info.roi == i)
        im = image.cut_image(components_mark, replace_mask=True)
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

        # print(f"[run] {im}")
        ImageWriter.save(im, os.path.join(dir_path, f"{file_name}_component{i}.tif"))
        step_changed(2 * i + 1)
        ImageWriter.save_mask(im, os.path.join(dir_path, f"{file_name}_component{i}_mask.tif"))
        step_changed(2 * i + 2)


class SaveComponents(SaveBase):
    """
    Save selected components in separated files.
    """

    @classmethod
    def get_short_name(cls):
        return "comp"

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info: MaskProjectTuple,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        save_components(
            project_info.image,
            project_info.selected_components,
            save_location,
            project_info.roi_info,
            project_info.points,
            range_changed,
            step_changed,
        )

    @classmethod
    def get_name(cls) -> str:
        return "Components"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []


class SaveParametersJSON(SaveBase):
    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict = None,
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
        with open(save_location, "w") as ff:
            json.dump({"parameters": project_info.roi_extraction_parameters}, ff, cls=ProfileEncoder)

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
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
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
    return UpdateLoadedMetadataMask.load_json_data(data)


class UpdateLoadedMetadataMask(UpdateLoadedMetadataBase):
    @classmethod
    def update_segmentation_profile(cls, profile_data: ROIExtractionProfile) -> ROIExtractionProfile:
        profile_data = super().update_segmentation_profile(profile_data)
        if profile_data.algorithm in {"Threshold", "Auto Threshold"}:
            if isinstance(profile_data.values["smooth_border"], bool):
                if profile_data.values["smooth_border"]:
                    profile_data.values["smooth_border"] = {
                        "name": "Opening",
                        "values": {"smooth_border_radius": profile_data.values["smooth_border_radius"]},
                    }
                else:
                    profile_data.values["smooth_border"] = {"name": "None", "values": {}}
                if "smooth_border_radius" in profile_data.values:
                    del profile_data.values["smooth_border_radius"]
            if "noise_removal" in profile_data.values:
                profile_data.values["noise_filtering"] = profile_data.values["noise_removal"]
                del profile_data.values["noise_removal"]
        return profile_data


load_dict = Register(
    LoadStackImage, LoadROIImage, LoadStackImageWithMask, LoadPoints, class_methods=LoadBase.need_functions
)
save_parameters_dict = Register(SaveParametersJSON, class_methods=SaveBase.need_functions)
save_components_dict = Register(SaveComponents, class_methods=SaveBase.need_functions)
save_segmentation_dict = Register(
    SaveROI, SaveMaskAsTiff, SaveROIAsTIFF, SaveROIAsNumpy, class_methods=SaveBase.need_functions
)
