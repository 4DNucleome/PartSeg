import json
import os
import sys
import tarfile
import typing
from copy import copy
from functools import partial
from io import BufferedIOBase, BytesIO, IOBase, RawIOBase, TextIOBase
from pathlib import Path
from threading import Lock

import numpy as np
import tifffile
from packaging.version import Version
from packaging.version import parse as parse_version
from tifffile import TiffFile

from PartSegImage import GenericImageReader

from ..algorithm_describe_base import Register, ROIExtractionProfile
from ..io_utils import (
    LoadBase,
    LoadPoints,
    SegmentationType,
    UpdateLoadedMetadataBase,
    WrongFileTypeException,
    check_segmentation_type,
    open_tar_file,
    proxy_callback,
    tar_to_buff,
)
from ..mask.io_functions import LoadROIImage
from ..project_info import HistoryElement
from ..roi_info import ROIInfo
from ..universal_const import UNIT_SCALE, Units
from .analysis_utils import SegmentationPipeline, SegmentationPipelineElement
from .calculation_plan import CalculationPlan, CalculationTree, MeasurementCalculate
from .io_utils import MaskInfo, ProjectTuple, project_version_info
from .measurement_base import Leaf, MeasurementEntry, Node
from .measurement_calculation import MeasurementProfile
from .save_hooks import part_hook

__all__ = [
    "LoadStackImage",
    "LoadImageMask",
    "LoadProject",
    "LoadMask",
    "load_dict",
    "load_metadata",
    "UpdateLoadedMetadataAnalysis",
    "LoadMaskSegmentation",
]


def load_project(
    file: typing.Union[str, Path, tarfile.TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase]
) -> ProjectTuple:
    """Load project from archive"""
    tar_file, file_path = open_tar_file(file)
    try:
        if check_segmentation_type(tar_file) != SegmentationType.analysis:
            raise WrongFileTypeException()
        image_buffer = BytesIO()
        image_tar = tar_file.extractfile(tar_file.getmember("image.tif"))
        image_buffer.write(image_tar.read())
        image_buffer.seek(0)
        reader = GenericImageReader()
        image = reader.read(image_buffer, ext=".tif")
        image.file_path = file_path

        algorithm_str = tar_file.extractfile("algorithm.json").read()
        algorithm_dict = load_metadata(algorithm_str)
        algorithm_dict = update_algorithm_dict(algorithm_dict)
        algorithm_dict.get("project_file_version")
        metadata = json.loads(tar_file.extractfile("metadata.json").read())
        try:
            version = parse_version(metadata["project_version_info"])
        except KeyError:
            version = Version("1.0")
        if version == Version("1.0"):
            seg_dict = np.load(tar_to_buff(tar_file, "segmentation.npz"))
            mask = seg_dict["mask"] if "mask" in seg_dict else None
            roi = seg_dict["segmentation"]
        else:
            roi = tifffile.imread(tar_to_buff(tar_file, "segmentation.tif"))
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
        history = []
        try:
            history_buff = tar_file.extractfile(tar_file.getmember("history/history.json")).read()
            history_json = load_metadata(history_buff)
            for el in history_json:
                history_buffer = BytesIO()
                history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
                history_buffer.seek(0)
                el = update_algorithm_dict(el)
                segmentation_parameters = {"algorithm_name": el["algorithm_name"], "values": el["values"]}
                history.append(
                    HistoryElement(
                        roi_extraction_parameters=segmentation_parameters,
                        mask_property=el["mask_property"],
                        arrays=history_buffer,
                        annotations=el.get("annotations", {}),
                    )
                )

        except KeyError:
            pass
    finally:
        if isinstance(file, (str, Path)):
            tar_file.close()
    image.set_mask(mask)
    roi_info = ROIInfo(roi, annotations=metadata.get("roi_annotations"), alternative=alternative)
    if version <= project_version_info:
        return ProjectTuple(
            file_path=file_path,
            image=image,
            roi_info=roi_info,
            mask=mask,
            history=history,
            algorithm_parameters=algorithm_dict,
        )

    print("This project is from new version of PartSeg:", version, project_version_info, file=sys.stderr)
    return ProjectTuple(
        file_path=file_path,
        image=image,
        roi_info=roi_info,
        mask=mask,
        history=history,
        algorithm_parameters=algorithm_dict,
        errors="This project is from new version of PartSeg. It may load incorrect.",
    )


class LoadProject(LoadBase):
    @classmethod
    def get_name(cls):
        return "Project (*.tgz *.tbz2 *.gz *.bz2)"

    @classmethod
    def get_short_name(cls):
        return "project"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> ProjectTuple:
        return load_project(load_locations[0])


class LoadStackImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image (*.tif *.tiff *.lsm *.czi *.oib *.oif)"

    @classmethod
    def get_short_name(cls):
        return "tiff_image"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ):
        if metadata is None:
            metadata = {"default_spacing": tuple(1 / UNIT_SCALE[Units.nm.value] for _ in range(3))}

        if "recursion_limit" not in metadata:
            metadata = copy(metadata)
            metadata["recursion_limit"] = 3
        image = GenericImageReader.read_image(
            load_locations[0],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=tuple(metadata["default_spacing"]),
        )
        re_read = all(el[0] == el[1] for el in image.get_ranges())
        if re_read and metadata["recursion_limit"] > 0:
            metadata["recursion_limit"] -= 1
            cls.load(load_locations, range_changed, step_changed, metadata)
        return ProjectTuple(load_locations[0], image)


class LoadImageMask(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image with mask (*.tif *.tiff *.lsm)"

    @classmethod
    def get_short_name(cls):
        return "image_with_mask"

    @classmethod
    def number_of_files(cls):
        return 2

    @classmethod
    def correct_files_order(cls, paths):
        name1, name2 = [os.path.basename(os.path.splitext(x)[0]) for x in paths]
        if name2.endswith("_mask"):
            return [name1, name2]
        return paths

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ):
        if metadata is None:
            metadata = {"default_spacing": (10 ** -6, 10 ** -6, 10 ** -6)}
        if len(load_locations) == 1:
            new_path, ext = os.path.splitext(load_locations[0])
            new_path += "_mask" + ext
            if not os.path.exists(new_path):
                raise ValueError("Cannot determine mask file. It need to have '_mask' suffix.")
            load_locations.append(load_locations)
        image = GenericImageReader.read_image(
            load_locations[0],
            load_locations[1],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=tuple(metadata["default_spacing"]),
        )
        return ProjectTuple(load_locations[0], image, mask=image.mask)

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        base, ext = os.path.splitext(file_paths[0])
        return base + "_mask" + ext


class LoadMask(LoadBase):
    @classmethod
    def get_name(cls):
        return "mask to image (*.tif *.tiff)"

    @classmethod
    def get_short_name(cls):
        return "mask_to_name"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ):
        image_file = TiffFile(load_locations[0])
        count_pages = [0]
        mutex = Lock()

        def report_func():
            mutex.acquire()
            count_pages[0] += 1
            step_changed(count_pages[0])
            mutex.release()

        range_changed(0, len(image_file.series[0]))
        image_file.report_func = report_func
        mask_data = image_file.asarray()
        return MaskInfo(load_locations[0], mask_data)

    @classmethod
    def partial(cls):
        return True


class LoadMaskSegmentation(LoadBase):
    @classmethod
    def get_name(cls):
        return "mask project (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "mask_project"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.List[ProjectTuple]:
        data = LoadROIImage.load(load_locations, range_changed, step_changed, metadata)
        image = data.image
        roi = data.roi_info.roi
        components = data.selected_components
        res = []
        base, ext = os.path.splitext(load_locations[0])
        path_template = base + "_component{}" + ext
        for i in components:
            single_roi = roi == i
            if not np.any(single_roi):
                continue
            im = image.cut_image(roi == i, replace_mask=True)
            im.file_path = path_template.format(i)
            res.append(ProjectTuple(im.file_path, im, mask=im.mask))
        return res


class UpdateLoadedMetadataAnalysis(UpdateLoadedMetadataBase):
    json_hook = part_hook

    @classmethod
    def update_segmentation_profile(cls, profile_data: ROIExtractionProfile) -> ROIExtractionProfile:
        replace_name_dict = {
            "Split Mask on Part": "Mask Distance Splitting",
            "Lower threshold flow": "Lower threshold with watershed",
            "Upper threshold flow": "Upper threshold with watershed",
        }
        if profile_data.algorithm in replace_name_dict:
            profile_data.algorithm = replace_name_dict[profile_data.algorithm]
        return super().update_segmentation_profile(profile_data)

    @classmethod
    def update_calculation_tree(cls, data: CalculationTree) -> CalculationTree:
        data.operation = cls.recursive_update(data.operation)
        data.children = [cls.update_calculation_tree(x) for x in data.children]
        return data

    @classmethod
    def update_calculation_plan(cls, data: CalculationPlan) -> CalculationPlan:
        data.execution_tree = cls.update_calculation_tree(data.execution_tree)
        return data

    @classmethod
    def update_segmentation_pipeline_element(cls, data: SegmentationPipelineElement) -> SegmentationPipelineElement:
        return SegmentationPipelineElement(cls.update_segmentation_profile(data.segmentation), data.mask_property)

    @classmethod
    def update_segmentation_pipeline(cls, data: SegmentationPipeline) -> SegmentationPipeline:
        return SegmentationPipeline(
            data.name,
            cls.update_segmentation_profile(data.segmentation),
            [cls.update_segmentation_pipeline_element(x) for x in data.mask_history],
        )

    @classmethod
    def update_measurement_profile(cls, data: MeasurementProfile) -> MeasurementProfile:
        data.chosen_fields = [cls.update_measurement_entry(x) for x in data.chosen_fields]
        return data

    @classmethod
    def update_measurement_entry(cls, data: MeasurementEntry) -> MeasurementEntry:
        return data.replace_(calculation_tree=cls.update_measurement_calculation_tree(data.calculation_tree))

    @classmethod
    def update_measurement_calculation_tree(cls, data: typing.Union[Leaf, Node]) -> typing.Union[Leaf, Node]:
        if isinstance(data, Leaf):
            replace_name_dict = {
                "Moment of inertia": "Moment",
                "Components Number": "Components number",
                "Pixel Brightness Sum": "Pixel brightness sum",
                "Longest main axis length": "First principal axis length",
                "Middle main axis length": "Second principal axis length",
                "Shortest main axis length": "Third principal axis length",
                "split on part volume": "distance splitting volume",
                "split on part pixel brightness sum": "distance splitting pixel brightness sum",
                "Rim Volume": "rim volume",
                "Rim Pixel Brightness Sum": "rim pixel brightness sum",
            }

            if data.name in replace_name_dict:
                # noinspection PyUnresolvedReferences
                return data._replace(name=replace_name_dict[data.name])
            return data

        return data.replace_(
            left=cls.update_measurement_calculation_tree(data.left),
            right=cls.update_measurement_calculation_tree(data.right),
        )

    @classmethod
    def update_measurement_calculate(cls, data: MeasurementCalculate):
        return data.replace_(measurement_profile=cls.update_measurement_profile(data.measurement_profile))

    @classmethod
    def recursive_update(cls, data):
        if isinstance(data, CalculationPlan):
            return cls.update_calculation_plan(data)
        if isinstance(data, CalculationTree):
            return cls.update_calculation_tree(data)
        if isinstance(data, SegmentationPipeline):
            return cls.update_segmentation_pipeline(data)
        if isinstance(data, MeasurementProfile):
            return cls.update_measurement_profile(data)
        if isinstance(data, MeasurementCalculate):
            return cls.update_measurement_calculate(data)

        return super().recursive_update(data)


def load_metadata(data: typing.Union[str, Path]):
    """
    Load metadata saved in json format for segmentation mask

    :param data: path to json file, string with json, or opened file
    :return: restored structures
    """
    return UpdateLoadedMetadataAnalysis.load_json_data(data)


def update_algorithm_dict(dkt):
    if "name" in dkt:
        profile = ROIExtractionProfile("", dkt["name"], dkt["values"])
    elif "algorithm_name" in dkt:
        profile = ROIExtractionProfile("", dkt["algorithm_name"], dkt["values"])
    else:
        return dkt
    profile = UpdateLoadedMetadataAnalysis.recursive_update(profile)
    res = dict(dkt)
    res.update({"algorithm_name": profile.algorithm, "values": profile.values})
    return res


load_dict = Register(
    LoadStackImage, LoadImageMask, LoadProject, LoadMaskSegmentation, LoadPoints, class_methods=LoadBase.need_functions
)
