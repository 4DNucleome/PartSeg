import json
import os
import re
import typing
from abc import ABC
from datetime import datetime
from enum import Enum
from io import BytesIO, StringIO
from pathlib import Path
from tarfile import TarInfo, TarFile

from PartSeg.tiff_image import Image
from PartSeg.utils.json_hooks import profile_hook
from .algorithm_describe_base import AlgorithmDescribeBase, SegmentationProfile


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
    elif "metadata.json" in names:
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


class ProjectInfoBase:
    file_path: str
    image: Image

    def _replace(self, file_path=None, image=None):
        pass

    def get_raw_copy(self):
        raise NotImplementedError

    def is_raw(self):
        raise NotImplementedError


class SaveBase(AlgorithmDescribeBase, ABC):
    need_functions = ["save", "get_short_name", "get_name_with_suffix", "get_default_extension",
                      "need_segmentation", "need_mask"]
    @classmethod
    def get_short_name(cls):
        raise NotImplementedError()

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict,
             range_changed=None, step_changed=None):
        raise NotImplementedError()

    @classmethod
    def get_name_with_suffix(cls):
        return cls.get_name()

    @classmethod
    def get_default_extension(cls):
        match = re.search(r'\(\*(\.\w+)', cls.get_name_with_suffix())
        if match:
            return match.group(1)
        else:
            return ""

    @classmethod
    def need_segmentation(cls):
        return True

    @classmethod
    def need_mask(cls):
        return False


class LoadBase(AlgorithmDescribeBase, ABC):
    need_functions = ["load", "get_short_name", "get_name_with_suffix", "number_of_files", "correct_files_order",
                      "get_next_file", "partial"]

    @classmethod
    def get_short_name(cls):
        raise NotImplementedError()

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        raise NotImplementedError()

    @classmethod
    def get_name_with_suffix(cls):
        return cls.get_name()

    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def get_extensions(cls) -> typing.List:
        match = re.match(r'.*\((.*)\)', cls.get_name())
        if match:
            return [x[1:] for x in match.group(1).split(' ') if len(x) > 3]
        else:
            return []

    @classmethod
    def number_of_files(cls):
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
        if isinstance(data, typing.TextIO):
            decoded_data = json.load(data, object_hook=cls.json_hook)
        elif os.path.exists(data):
            with open(data, "r") as ff:
                decoded_data = json.load(ff, object_hook=cls.json_hook)
        else:
            decoded_data = json.loads(data, object_hook=cls.json_hook)
        return cls.recursive_update(decoded_data)

    @classmethod
    def recursive_update(cls, data):
        if isinstance(data, (tuple, list)):
            return type(data)([cls.recursive_update(x) for x in data])
        if isinstance(data, SegmentationProfile):
            return cls.update_segmentation_profile(data)
        if isinstance(data, Enum):
            return cls.update_enum(data)
        return data

    @classmethod
    def update_enum(cls, enum_data: Enum):
        return enum_data

    # noinspection PyUnusedLocal
    @classmethod
    def update_segmentation_sub_dict(cls, name: str, dkt: dict) -> dict:
        for key in dkt["values"].keys():
            item = dkt["values"][key]
            if isinstance(item, Enum):
                dkt["values"][key] = cls.update_enum(item)
            elif isinstance(item, dict):
                dkt["values"][key] = cls.update_segmentation_sub_dict(key, item)
        return dkt

    @classmethod
    def update_segmentation_profile(cls, profile_data: SegmentationProfile) -> SegmentationProfile:
        for key in profile_data.values.keys():
            item = profile_data.values[key]
            if isinstance(item, Enum):
                profile_data.values[key] = cls.update_enum(item)
            elif isinstance(item, dict):
                profile_data.values[key] = cls.update_segmentation_sub_dict(key, item)
        return profile_data


def load_metadata_base(data: typing.Union[str, Path]):
    return UpdateLoadedMetadataBase.load_json_data(data)


def proxy_callback(range_changed: typing.Callable[[int, int], typing.Any],
                   step_changed: typing.Callable[[int], typing.Any], text: str, val):
    if text == "max" and range_changed is not None:
        range_changed(0, val)
    if text == "step" and step_changed is not None:
        step_changed(val)
