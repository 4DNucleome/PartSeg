import os
import tarfile
import typing
from copy import copy
from functools import partial
from io import TextIOBase, BufferedIOBase, RawIOBase, IOBase, BytesIO
from pathlib import Path
from threading import Lock
import numpy as np
import json
from tifffile import TiffFile
import packaging.version

from PartSegImage import ImageReader
from PartSeg.utils.analysis.calculation_plan import CalculationPlan, CalculationTree
from PartSeg.utils.universal_const import Units, UNIT_SCALE
from ..algorithm_describe_base import Register, SegmentationProfile
from .analysis_utils import HistoryElement, SegmentationPipeline, SegmentationPipelineElement
from .io_utils import ProjectTuple, MaskInfo, project_version_info
from .save_hooks import part_hook
from ..io_utils import LoadBase, proxy_callback, check_segmentation_type, SegmentationType, WrongFileTypeException, \
    UpdateLoadedMetadataBase

__all__ = ["LoadImage", "LoadImageMask", "LoadProject", "LoadMask", "load_dict"]


def load_project(
        file: typing.Union[str, tarfile.TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase]) -> ProjectTuple:
    """Load project from archive"""
    if isinstance(file, tarfile.TarFile):
        tar_file = file
        file_path = ""
    elif isinstance(file, str):
        tar_file = tarfile.open(file)
        file_path = file
    elif isinstance(file, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
        tar_file = tarfile.open(fileobj=file)
        file_path = ""
    else:
        raise ValueError(f"wrong type of file_ argument: {type(file)}")
    if check_segmentation_type(tar_file) != SegmentationType.analysis:
        raise WrongFileTypeException()
    image_buffer = BytesIO()
    image_tar = tar_file.extractfile(tar_file.getmember("image.tif"))
    image_buffer.write(image_tar.read())
    image_buffer.seek(0)
    reader = ImageReader()
    image = reader.read(image_buffer)
    image.file_path = file_path
    seg_tar = tar_file.extractfile(tar_file.getmember("segmentation.npz"))
    seg_buffer = BytesIO()
    seg_buffer.write(seg_tar.read())
    seg_buffer.seek(0)
    seg_dict = np.load(seg_buffer)
    if "mask" in seg_dict:
        mask = seg_dict["mask"]
    else:
        mask = None
    algorithm_str = tar_file.extractfile("algorithm.json").read()
    algorithm_dict = load_metadata(algorithm_str)
    algorithm_dict = update_algorithm_dict(algorithm_dict)
    algorithm_dict.get("project_file_version")
    try:
        version = packaging.version.parse(json.loads(tar_file.extractfile("metadata.json").read()))
    except KeyError:
        version = packaging.version.Version("1.0")
    history = []
    try:
        history_buff = tar_file.extractfile(tar_file.getmember("history/history.json"))
        history_json = load_metadata(history_buff)
        for el in history_json:
            history_buffer = BytesIO()
            history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
            history_buffer.seek(0)
            el = update_algorithm_dict(el)
            history.append(HistoryElement(algorithm_name=el["algorithm_name"], algorithm_values=el["values"],
                                          mask_property=el["mask_property"], arrays=history_buffer))

    except KeyError:
        pass
    if isinstance(file, str):
        tar_file.close()
    if version >= project_version_info:
        return ProjectTuple(file_path, image, seg_dict["segmentation"], seg_dict["full_segmentation"], mask, history,
                            algorithm_dict)
    else:
        print(version, project_version_info)
        return ProjectTuple(file_path, image, seg_dict["segmentation"], seg_dict["full_segmentation"], mask, history,
                            algorithm_dict, "This project is from new version of PartSeg. It may load incorrect.")


class LoadProject(LoadBase):
    @classmethod
    def get_name(cls):
        return "Project (*.tgz *.tbz2 *.gz *.bz2)"

    @classmethod
    def get_short_name(cls):
        return "project"

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        return load_project(load_locations[0])


class LoadImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image (*.tif *.tiff *.lsm)"

    @classmethod
    def get_short_name(cls):
        return "tiff_image"

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        if metadata is None:
            metadata = {"default_spacing": [1 / UNIT_SCALE[Units.nm.value] for _ in range(3)]}
        if "recursion_limit" not in metadata:
            metadata = copy(metadata)
            metadata["recursion_limit"] = 3
        image = ImageReader.read_image(
            load_locations[0], callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
        re_read = True
        for el in image.get_ranges():
            if el[0] != el[1]:
                re_read = False
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
        else:
            return paths

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        if metadata is None:
            metadata = {"default_spacing": [1, 1, 1]}
        image = ImageReader.read_image(
            load_locations[0], load_locations[1],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
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
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
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


class UpdateLoadedMetadataAnalysis(UpdateLoadedMetadataBase):
    json_hook = part_hook

    @classmethod
    def update_calculation_tree(cls, data: CalculationTree):
        data.operation = cls.recursive_update(data.operation)
        data.children = [cls.update_calculation_tree(x) for x in data.children]
        return data

    @classmethod
    def update_calculation_plan(cls, data: CalculationPlan):
        data.execution_tree = cls.update_calculation_tree(data.execution_tree)
        return data

    @classmethod
    def update_segmentation_pipeline_element(cls, data: SegmentationPipelineElement):
        return SegmentationPipelineElement(cls.update_segmentation_profile(data.segmentation),
                                           data.mask_property)

    @classmethod
    def update_segmentation_pipeline(cls, data: SegmentationPipeline):
        return SegmentationPipeline(
            data.name, cls.update_segmentation_profile(data.segmentation),
            [cls.update_segmentation_pipeline_element(x) for x in data.mask_history]
        )

    @classmethod
    def recursive_update(cls, data):
        if isinstance(data, CalculationPlan):
            return cls.update_calculation_plan(data)
        if isinstance(data, CalculationTree):
            return cls.update_calculation_tree(data)
        if isinstance(data, SegmentationPipeline):
            return cls.update_segmentation_pipeline(data)

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
        profile = SegmentationProfile("", dkt["name"], dkt["values"])
    else:
        profile = SegmentationProfile("", dkt["algorithm_name"], dkt["values"])
    profile = UpdateLoadedMetadataAnalysis.recursive_update(profile)
    res = dict(dkt)
    res.update({"algorithm_name": profile.algorithm, "values": profile.values})
    return res


load_dict = Register(LoadImage, LoadImageMask, LoadProject, class_methods=LoadBase.need_functions)
