import json
import os
import tarfile
import typing
from functools import partial
from io import TextIOBase, BufferedIOBase, RawIOBase, IOBase, BytesIO
from pathlib import Path
from threading import Lock
import numpy as np
from tifffile import TiffFile

from PartSeg.tiff_image import ImageReader
from ..algorithm_describe_base import Register
from .analysis_utils import HistoryElement
from .io_utils import ProjectTuple, MaskInfo
from .save_hooks import part_hook
from ..io_utils import LoadBase, proxy_callback


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
    algorithm_dict = json.loads(algorithm_str, object_hook=part_hook)
    history = []
    try:
        history_buff = tar_file.extractfile(tar_file.getmember("history/history.json"))
        history_json = json.load(history_buff, object_hook=part_hook)
        for el in history_json:
            history_buffer = BytesIO()
            history_buffer.write(tar_file.extractfile(f"history/arrays_{el['index']}.npz").read())
            history_buffer.seek(0)
            history.append(HistoryElement(algorithm_name=el["algorithm_name"], algorithm_values=el["values"],
                                          mask_property=el["mask_property"], arrays=history_buffer))

    except KeyError:
        pass
    if isinstance(file, str):
        tar_file.close()
    return ProjectTuple(file_path, image, seg_dict["segmentation"], seg_dict["full_segmentation"], mask, history,
                        algorithm_dict)


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
            metadata = {"default_spacing": [1,1,1]}
        image = ImageReader.read_image(
            load_locations[0], callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
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
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        image = ImageReader.read_image(
            load_locations[0], load_locations[1],
            callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
        return ProjectTuple(load_locations[0], image)

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


load_dict = Register(LoadImage, LoadImageMask, LoadProject, LoadMask)
