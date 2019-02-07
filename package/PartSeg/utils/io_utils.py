import re
import typing
from abc import ABC
from datetime import datetime
from io import BytesIO, StringIO
from pathlib import Path
from tarfile import TarInfo

from ..utils.segmentation.algorithm_describe_base import AlgorithmDescribeBase


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
    @classmethod
    def get_short_name(cls):
        raise NotImplementedError()

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info, parameters: dict):
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
             callback_function: typing.Optional[typing.Callable] = None, default_spacing: typing.List[int]=None):
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
