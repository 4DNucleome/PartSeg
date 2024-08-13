import contextlib
import json
import logging
import os
import tarfile
import typing
from contextlib import suppress
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

from PartSegCore.algorithm_describe_base import Register, ROIExtractionProfile
from PartSegCore.analysis import AnalysisAlgorithmSelection
from PartSegCore.analysis.io_utils import MaskInfo, ProjectTuple, project_version_info
from PartSegCore.io_utils import (
    IO_MASK_METADATA_FILE,
    LoadBase,
    LoadPoints,
    SegmentationType,
    WrongFileTypeException,
    check_segmentation_type,
    load_metadata_base,
    load_metadata_part,
    open_tar_file,
    proxy_callback,
    tar_to_buff,
)
from PartSegCore.json_hooks import partseg_object_hook
from PartSegCore.mask.io_functions import LoadROIImage
from PartSegCore.project_info import HistoryElement
from PartSegCore.roi_info import ROIInfo
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import GenericImageReader

__all__ = [
    "LoadStackImage",
    "LoadImageMask",
    "LoadProject",
    "LoadMask",
    "load_dict",
    "load_metadata",
    "LoadMaskSegmentation",
    "LoadProfileFromJSON",
    "LoadImageForBatch",
]

from PartSegImage.image import Image


def _load_history(tar_file):
    history = []
    with suppress(KeyError):
        history_buff = tar_file.extractfile(tar_file.getmember("history/history.json")).read()
        history_json = load_metadata(history_buff)
        for el in history_json:
            history_buffer = BytesIO()
            history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
            history_buffer.seek(0)
            el_up = update_algorithm_dict(el)
            segmentation_parameters = {"algorithm_name": el_up["algorithm_name"], "values": el_up["values"]}
            history.append(
                HistoryElement(
                    roi_extraction_parameters=segmentation_parameters,
                    mask_property=el_up["mask_property"],
                    arrays=history_buffer,
                    annotations=el_up.get("annotations", {}),
                )
            )
    return history


def load_project_from_tar(tar_file, file_path):
    if check_segmentation_type(tar_file) != SegmentationType.analysis:
        raise WrongFileTypeException
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
    with contextlib.suppress(KeyError):
        algorithm_dict["algorithm_name"] = AnalysisAlgorithmSelection[algorithm_dict["algorithm_name"]].get_name()

    metadata = json.loads(tar_file.extractfile(IO_MASK_METADATA_FILE).read(), object_hook=partseg_object_hook)

    version = parse_version(metadata.get("project_version_info", "1.0"))

    if version == Version("1.0"):
        seg_dict = np.load(tar_to_buff(tar_file, "segmentation.npz"))
        mask = seg_dict.get("mask")
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
    history = _load_history(tar_file)
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
    logging.warning(
        "This project %(proj_ver) is from new version of PartSeg: %(version)",
        extra={"version": version, "proj_ver": project_version_info},
    )
    return ProjectTuple(
        file_path=file_path,
        image=image,
        roi_info=roi_info,
        mask=mask,
        history=history,
        algorithm_parameters=algorithm_dict,
        errors="This project is from new version of PartSeg. It may load incorrect.",
    )


def load_project(
    file: typing.Union[str, Path, tarfile.TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase]
) -> ProjectTuple:
    """Load project from archive"""
    tar_file, file_path = open_tar_file(file)
    try:
        return load_project_from_tar(tar_file, file_path)
    finally:
        if isinstance(file, (str, Path)):
            tar_file.close()


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
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> ProjectTuple:
        return load_project(load_locations[0])


class LoadStackImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image (*.tif *.tiff *.lsm *.czi *.oib *.oif *.obsep)"

    @classmethod
    def get_short_name(cls):
        return "tiff_image"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
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
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ):
        if metadata is None:
            metadata = {"default_spacing": (10**-6, 10**-6, 10**-6)}
        if len(load_locations) == 1:
            new_path, ext = os.path.splitext(load_locations[0])
            new_path += f"_mask{ext}"
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
        return f"{base}_mask{ext}"


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
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
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


def _mask_data_outside_mask(file_path):
    if not isinstance(file_path, str):
        return False
    with tarfile.open(file_path, "r:*") as tar_file:
        metadata = load_metadata_base(tar_file.extractfile(IO_MASK_METADATA_FILE).read().decode("utf8"))
        return metadata.get("keep_data_outside_mask", False)


def load_mask_project(
    load_locations: typing.List[typing.Union[str, BytesIO, Path]],
    range_changed: typing.Callable[[int, int], typing.Any],
    step_changed: typing.Callable[[int], typing.Any],
    metadata: typing.Optional[dict] = None,
):
    data = LoadROIImage.load(load_locations, range_changed, step_changed, metadata)
    zero_out_cut_area = _mask_data_outside_mask(load_locations[0])
    image = data.image
    if not isinstance(image, Image):  # pragma: no cover
        raise ValueError("Image is not instance of Image class.")
    if data.roi_info.roi is None:  # pragma: no cover
        raise ValueError("No ROI found in the image.")
    roi = data.roi_info.roi
    components = data.selected_components
    if not components:
        components = list(data.roi_info.bound_info)
    res = []
    base, ext = os.path.splitext(load_locations[0])
    range_changed(0, len(components))
    str_len = str(len(str(len(components))))
    path_template = base + "_component{:0" + str_len + "d}" + ext
    for i in components:
        step_changed(i)
        bound = data.roi_info.bound_info[i]
        single_roi = roi[tuple(bound.get_slices(data.frame_thickness))] == i
        if not np.any(single_roi):
            continue
        im = image.cut_image(
            bound.get_slices(), replace_mask=True, zero_out_cut_area=zero_out_cut_area, frame=data.frame_thickness
        ).cut_image(
            single_roi,
            replace_mask=True,
            zero_out_cut_area=zero_out_cut_area,
            frame=data.frame_thickness,
        )
        im.file_path = path_template.format(i)
        res.append(ProjectTuple(im.file_path, im, mask=im.mask))
    return res


class LoadMaskSegmentation(LoadBase):
    @classmethod
    def get_name(cls):
        return "Mask project (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "mask_project"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.List[ProjectTuple]:
        if range_changed is None:

            def range_changed(_x, _y):
                return None

        if step_changed is None:

            def step_changed(_):
                return None

        return load_mask_project(load_locations, range_changed, step_changed, metadata)


class LoadProfileFromJSON(LoadBase):
    @classmethod
    def get_short_name(cls):
        return "json"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Tuple[dict, list]:
        return load_metadata_part(load_locations[0])

    @classmethod
    def get_name(cls) -> str:
        return "Segment profile (*.json)"


def load_metadata(data: typing.Union[str, Path]):
    """
    Load metadata saved in json format for segmentation mask

    :param data: path to json file, string with json, or opened file
    :return: restored structures
    """
    return load_metadata_base(data)


def update_algorithm_dict(dkt):
    if "name" in dkt:
        profile = ROIExtractionProfile(name="", algorithm=dkt["name"], values=dkt["values"])
    elif "algorithm_name" in dkt:
        profile = ROIExtractionProfile(name="", algorithm=dkt["algorithm_name"], values=dkt["values"])
    else:
        return dkt
    res = dict(dkt)
    res.update({"algorithm_name": profile.algorithm, "values": profile.values})
    return res


load_dict = Register(
    LoadStackImage,
    LoadImageMask,
    LoadProject,
    LoadMaskSegmentation,
    LoadPoints,
    class_methods=LoadBase.need_functions,
)


class LoadImageForBatch(LoadBase):
    @classmethod
    def get_short_name(cls):
        return "load_all"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Optional[typing.Callable[[int, int], typing.Any]] = None,
        step_changed: typing.Optional[typing.Callable[[int], typing.Any]] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectTuple, typing.List[ProjectTuple]]:
        ext = os.path.splitext(load_locations[0])[1].lower()

        for loader in load_dict.values():
            if loader.partial() or loader.number_of_files() != 1:
                continue
            if ext in loader.get_extensions():
                res = loader.load([load_locations[0]], metadata=metadata)
                if isinstance(res, list):
                    return [cls._clean_project(x) for x in res]
                return cls._clean_project(res)
        raise ValueError(f"Cannot load file {load_locations[0]}")

    @staticmethod
    def _clean_project(project: ProjectTuple):
        return ProjectTuple(file_path=project.file_path, image=project.image.substitute(mask=None))

    @classmethod
    def get_name(cls) -> str:
        ext_set = set()
        for loader in load_dict.values():
            if loader.partial() or loader.number_of_files() != 1:
                continue
            ext_set.update(loader.get_extensions())

        return f"Load generic ({' '.join(f'*{x}' for x in ext_set)})"
