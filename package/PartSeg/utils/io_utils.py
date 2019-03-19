import re
import typing
from abc import ABC
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from tarfile import TarInfo

from PartSeg.tiff_image import Image
from .algorithm_describe_base import AlgorithmDescribeBase


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
        return re.search(r'\(\*(\.\w+)', cls.get_name_with_suffix()).group(1)

    @classmethod
    def need_segmentation(cls):
        return True

    @classmethod
    def need_mask(cls):
        return False


class LoadBase(AlgorithmDescribeBase, ABC):
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
    def number_of_files(cls):
        return 1

    @classmethod
    def get_next_file(cls, file_paths: typing.List[str]):
        return file_paths[0]

    @classmethod
    def partial(cls):
        """Inform that this class load complete data"""
        return False


def proxy_callback(range_changed: typing.Callable[[int, int], typing.Any],
                   step_changed: typing.Callable[[int], typing.Any], text: str, val):
    if text == "max" and range_changed is not None:
        range_changed(0, val)
    if text == "step" and step_changed is not None:
        step_changed(val)
