import json
import os
import re
import typing
from abc import ABC
from datetime import datetime
from enum import Enum
from io import BufferedIOBase, BytesIO, IOBase, RawIOBase, StringIO, TextIOBase
from pathlib import Path
from tarfile import TarFile, TarInfo

import imageio
import numpy as np
import pandas as pd
import tifffile
from napari.utils import Colormap

from PartSegCore.json_hooks import ProfileDict, profile_hook
from PartSegImage import ImageWriter
from PartSegImage.image import minimal_dtype

from .algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from .project_info import ProjectInfoBase


class SegmentationType(Enum):
    analysis = 1
    mask = 2


class WrongFileTypeException(Exception):
    pass


class NotSupportedImage(Exception):
    pass


def check_segmentation_type(tar_file: TarFile) -> SegmentationType:
    names = [x.name for x in tar_file.getmembers()]
    if "algorithm.json" in names:
        return SegmentationType.analysis
    if "metadata.json" in names:
        return SegmentationType.mask
    raise WrongFileTypeException()


def get_tarinfo(name, buffer: typing.Union[BytesIO, StringIO]):
    tar_info = TarInfo(name=name)
    buffer.seek(0)
    if isinstance(buffer, BytesIO):
        tar_info.size = len(buffer.getbuffer())
    else:
        tar_info.size = len(buffer.getvalue())
    tar_info.mtime = datetime.now().timestamp()
    return tar_info


class SaveBase(AlgorithmDescribeBase, ABC):
    need_functions = [
        "save",
        "get_short_name",
        "get_name_with_suffix",
        "get_default_extension",
        "need_segmentation",
        "need_mask",
    ]

    @classmethod
    def get_short_name(cls):
        raise NotImplementedError()

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        """

        :param save_location: location to save, can be buffer
        :param project_info: all project data
        :param parameters: additional parameters for saving method
        :param range_changed: report function for inform about steps num
        :param step_changed: report function for progress
        """
        raise NotImplementedError()

    @classmethod
    def get_name_with_suffix(cls):
        return cls.get_name()

    @classmethod
    def get_default_extension(cls):
        match = re.search(r"\(\*(\.\w+)", cls.get_name_with_suffix())
        if match:
            return match.group(1)
        return ""

    @classmethod
    def need_segmentation(cls):
        return True

    @classmethod
    def need_mask(cls):
        return False

    @classmethod
    def get_extensions(cls) -> typing.List[str]:
        match = re.match(r".*\((.*)\)", cls.get_name())
        if match is None:
            raise ValueError(f"No extensions found in {cls.get_name()}")
        extensions = match.group(1).split(" ")
        if not all(x.startswith("*.") for x in extensions):
            raise ValueError(f"Error with parsing extensions in {cls.get_name()}")
        return [x[1:] for x in extensions]


class LoadBase(AlgorithmDescribeBase, ABC):
    need_functions = [
        "load",
        "get_short_name",
        "get_name_with_suffix",
        "number_of_files",
        "correct_files_order",
        "get_next_file",
        "partial",
    ]

    @classmethod
    def get_short_name(cls):
        raise NotImplementedError()

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> typing.Union[ProjectInfoBase, typing.List[ProjectInfoBase]]:
        """
        Function for load data

        :param load_locations: list of files to load
        :param range_changed: callback function for inform about number of steps to be done
        :param step_changed:  callback function for report that single step has been done
        :param metadata: additional information needed by function. Like default spacing for load image
        :return: Project info or list of project info
        """
        raise NotImplementedError()

    @classmethod
    def get_name_with_suffix(cls):
        return cls.get_name()

    @classmethod
    def get_extensions(cls) -> typing.List[str]:
        match = re.match(r".*\((.*)\)", cls.get_name())
        if match is None:
            raise ValueError(f"No extensions found in {cls.get_name()}")
        extensions = match.group(1).split(" ")
        if not all(x.startswith("*.") for x in extensions):
            raise ValueError(f"Error with parsing extensions in {cls.get_name()}")
        return [x[1:] for x in extensions]

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def number_of_files(cls):
        """Number of files required for load method"""
        return 1

    @classmethod
    def correct_files_order(cls, paths):
        return paths

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        return file_paths[0]

    @classmethod
    def partial(cls):
        """Inform that this class load complete data"""
        return False


class UpdateLoadedMetadataBase:
    json_hook = staticmethod(profile_hook)

    @classmethod
    def load_json_data(cls, data: typing.Union[str, Path, typing.TextIO]):
        try:
            if isinstance(data, typing.TextIO):
                decoded_data = json.load(data, object_hook=cls.json_hook)
            elif os.path.exists(data):
                with open(data) as ff:
                    decoded_data = json.load(ff, object_hook=cls.json_hook)
            else:
                decoded_data = json.loads(data, object_hook=cls.json_hook)
        except ValueError:
            decoded_data = json.loads(data, object_hook=cls.json_hook)
        return cls.recursive_update(decoded_data)

    @classmethod
    def recursive_update(cls, data):
        if isinstance(data, (tuple, list)):
            return type(data)([cls.recursive_update(x) for x in data])
        if isinstance(data, ROIExtractionProfile):
            return cls.update_segmentation_profile(data)
        if isinstance(data, Enum):
            return cls.update_enum(data)
        if isinstance(data, dict):
            for key in data.keys():
                data[key] = cls.recursive_update(data[key])
                if key == "custom_colormap":
                    cls.update_colormaps(data[key])
        if isinstance(data, ProfileDict):
            data.my_dict = cls.recursive_update(data.my_dict)
        return data

    @staticmethod
    def update_colormaps(dkt: dict):
        for key, val in dkt.items():
            if isinstance(val, Colormap):
                val.name = key

    @classmethod
    def update_enum(cls, enum_data: Enum):
        return enum_data

    # noinspection PyUnusedLocal
    @classmethod
    def update_segmentation_sub_dict(cls, name: str, dkt: dict) -> dict:
        if "values" not in dkt:
            return dkt
        if name == "sprawl_type" and dkt["name"].endswith(" sprawl"):
            dkt["name"] = dkt["name"][: -len(" sprawl")]
        for key in dkt["values"].keys():
            item = dkt["values"][key]
            if isinstance(item, Enum):
                dkt["values"][key] = cls.update_enum(item)
            elif isinstance(item, dict):
                dkt["values"][key] = cls.update_segmentation_sub_dict(key, item)
        return dkt

    @classmethod
    def update_segmentation_profile(cls, profile_data: ROIExtractionProfile) -> ROIExtractionProfile:
        for key in list(profile_data.values.keys()):
            item = profile_data.values[key]
            if isinstance(item, Enum):
                profile_data.values[key] = cls.update_enum(item)
            elif isinstance(item, dict):
                if key == "noise_removal":
                    del profile_data.values[key]
                    key = "noise_filtering"
                if "values" in item and "gauss_type" in item["values"]:
                    item["values"]["dimension_type"] = item["values"]["gauss_type"]
                    del item["values"]["gauss_type"]
                profile_data.values[key] = cls.update_segmentation_sub_dict(key, item)
        return profile_data


def load_metadata_base(data: typing.Union[str, Path]):
    return UpdateLoadedMetadataBase.load_json_data(data)


def proxy_callback(
    range_changed: typing.Callable[[int, int], typing.Any],
    step_changed: typing.Callable[[int], typing.Any],
    text: str,
    val,
):
    if text == "max" and range_changed is not None:
        range_changed(0, val)
    if text == "step" and step_changed is not None:
        step_changed(val)


def open_tar_file(
    file_data: typing.Union[str, Path, TarFile, TextIOBase, BufferedIOBase, RawIOBase, IOBase], mode="r"
) -> typing.Tuple[TarFile, str]:
    """Create tar file from path or buffer. If passed :py:class:`TarFile` then return it."""
    if isinstance(file_data, TarFile):
        tar_file = file_data
        file_path = ""
    elif isinstance(file_data, (str, Path)):
        tar_file = TarFile.open(file_data, mode)
        file_path = str(file_data)
    elif isinstance(file_data, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
        tar_file = TarFile.open(fileobj=file_data, mode="r")
        file_path = ""
    else:
        raise ValueError(f"wrong type of file_ argument: {type(file_data)}")
    return tar_file, file_path


class SaveMaskAsTiff(SaveBase):
    @classmethod
    def get_name(cls):
        return "Mask (*.tiff *.tif)"

    @classmethod
    def get_short_name(cls):
        return "mask_tiff"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def need_mask(cls):
        return True

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        if project_info.image.mask is None and project_info.mask is not None:
            ImageWriter.save_mask(project_info.image.substitute(mask=project_info.mask), save_location)
        ImageWriter.save_mask(project_info.image, save_location)


def tar_to_buff(tar_file, member_name) -> BytesIO:
    tar_value = tar_file.extractfile(tar_file.getmember(member_name))
    buffer = BytesIO()
    buffer.write(tar_value.read())
    buffer.seek(0)
    return buffer


class SaveScreenshot(SaveBase):
    @classmethod
    def get_short_name(cls):
        return "screenshot"

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        imageio.imsave(save_location, project_info)

    @classmethod
    def get_name(cls) -> str:
        return "Screenshot (*.png *.jpg *.jpeg)"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return []


class SaveROIAsTIFF(SaveBase):
    @classmethod
    def get_name(cls):
        return "ROI as tiff (*.tiff *.tif)"

    @classmethod
    def get_short_name(cls):
        return "roi_tiff"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict,
        range_changed=None,
        step_changed=None,
    ):
        roi = project_info.roi_info.roi
        roi_max = max(project_info.roi_info.bound_info)
        roi = roi.astype(minimal_dtype(roi_max))
        tifffile.imsave(save_location, roi)


class SaveROIAsNumpy(SaveBase):
    @classmethod
    def get_name(cls):
        return "ROI as numpy (*.npy)"

    @classmethod
    def get_short_name(cls):
        return "ROI_numpy"

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: dict = None,
        range_changed=None,
        step_changed=None,
    ):
        roi = project_info.roi_info.roi
        roi_max = max(project_info.roi_info.bound_info)
        roi = roi.astype(minimal_dtype(roi_max))
        np.save(save_location, roi)


class PointsInfo(typing.NamedTuple):
    file_path: str
    points: np.ndarray


class LoadPoints(LoadBase):
    @classmethod
    def get_short_name(cls):
        return "point_csv"

    @classmethod
    def load(
        cls,
        load_locations: typing.List[typing.Union[str, BytesIO, Path]],
        range_changed: typing.Callable[[int, int], typing.Any] = None,
        step_changed: typing.Callable[[int], typing.Any] = None,
        metadata: typing.Optional[dict] = None,
    ) -> PointsInfo:
        df = pd.read_csv(load_locations[0], delimiter=",", index_col=0)
        return PointsInfo(load_locations[0], df.to_numpy())

    @classmethod
    def get_name(cls) -> str:
        return "Points (*.csv)"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return ["text"]
