import os
import tarfile
import typing
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import json
from typing import Union
from io import BytesIO, TextIOBase, BufferedIOBase, RawIOBase, IOBase

from PartSeg.utils.analysis.save_hooks import PartEncoder, part_hook
from ..io_utils import get_tarinfo, SaveBase, LoadBase, proxy_callback, ProjectInfoBase
from ..algorithm_describe_base import AlgorithmProperty, Register, SegmentationProfile
from PartSeg.tiff_image import Image, ImageWriter, ImageReader


class SegmentationTuple(ProjectInfoBase, typing.NamedTuple):
    file_path: str
    image: Union[Image, str, None]
    segmentation: typing.Optional[np.ndarray] = None
    chosen_components: typing.List = []
    segmentation_parameters: typing.Dict[int, typing.Optional[SegmentationProfile]] = {}

    def get_raw_copy(self):
        return SegmentationTuple(self.file_path, self.image.substitute())

    def is_raw(self):
        return self.segmentation is None


def save_stack_segmentation(file: Union[tarfile.TarFile, str, TextIOBase, BufferedIOBase, RawIOBase, IOBase],
                            # segmentation: np.ndarray, list_of_components, base_file: Optional[str]=None,
                            segmentation_info: SegmentationTuple, parameters: dict,
                            range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 5)
    if isinstance(file, tarfile.TarFile):
        tar_file = file
    elif isinstance(file, str):
        if not os.path.exists(os.path.dirname(file)):
            os.makedirs(os.path.dirname(file))
        ext = os.path.splitext(file)[1]
        if ext.lower() in ['.bz2', ".tbz2"]:
            tar_mode = 'w:bz2'
        else:
            tar_mode = 'w:gz'
        tar_file = tarfile.open(file, tar_mode)
    elif isinstance(file, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
        tar_file = tarfile.open(fileobj=file, mode='w:gz')
    else:
        raise ValueError(f"wrong type of file_ argument: {type(file)}")
    step_changed(1)
    segmentation_buff = BytesIO()
    # noinspection PyTypeChecker
    np.save(segmentation_buff, segmentation_info.segmentation)
    segmentation_tar = get_tarinfo("segmentation.npy", segmentation_buff)
    tar_file.addfile(segmentation_tar, fileobj=segmentation_buff)
    step_changed(3)
    metadata = {"components": segmentation_info.chosen_components,
                "parameters": segmentation_info.segmentation_parameters, "shape": segmentation_info.segmentation.shape}
    if isinstance(segmentation_info.image, Image):
        file_path = segmentation_info.image.file_path
    elif isinstance(segmentation_info.image, str):
        file_path = segmentation_info.image
    else:
        file_path = ""
    if file_path != "":
        if parameters["relative_path"] and isinstance(file, str):
            metadata["base_file"] = os.path.relpath(file_path, os.path.dirname(file))
        else:
            metadata["base_file"] = file_path
    metadata_buff = BytesIO(json.dumps(metadata, cls=PartEncoder).encode('utf-8'))
    metadata_tar = get_tarinfo("metadata.json", metadata_buff)
    tar_file.addfile(metadata_tar, metadata_buff)
    step_changed(4)
    if isinstance(file, str):
        tar_file.close()
    step_changed(5)


def load_stack_segmentation(file: str, range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0, 4)
    if isinstance(file, tarfile.TarFile):
        tar_file = file
    elif isinstance(file, str):
        tar_file = tarfile.open(file)
    elif isinstance(file, (TextIOBase, BufferedIOBase, RawIOBase, IOBase)):
        tar_file = tarfile.open(fileobj=file)
    else:
        raise ValueError(f"wrong type of file_ argument: {type(file)}")
    step_changed(1)
    segmentation_buff = BytesIO()
    segmentation_tar = tar_file.extractfile(tar_file.getmember("segmentation.npy"))
    segmentation_buff.write(segmentation_tar.read())
    step_changed(2)
    segmentation_buff.seek(0)
    segmentation = np.load(segmentation_buff)
    step_changed(3)
    metadata = json.loads(tar_file.extractfile("metadata.json").read().decode("utf8"), object_hook=part_hook)
    step_changed(4)
    if isinstance(file, str):
        tar_file.close()
    return segmentation, metadata


def empty_fun(_a0=None, _a1=None):
    pass


class LoadSegmentation(LoadBase):
    @classmethod
    def get_name(cls):
        return "Segmentation to image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        segmentation, metadata = load_stack_segmentation(load_locations[0], range_changed=range_changed,
                                                         step_changed=step_changed)

        if "parameters" not in metadata:
            parameters = defaultdict(lambda: None)
        else:
            parameters = dict([(int(k), v) for k, v in metadata["parameters"].items()])

        return SegmentationTuple(load_locations[0], metadata["base_file"] if "base_file" in metadata else None,
                                 segmentation, metadata["components"], parameters)

    @classmethod
    def partial(cls):
        return False


class LoadSegmentationImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Segmentation with image (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg_img"

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        seg = LoadSegmentation.load(load_locations)
        base_file = seg.image
        if base_file is None:
            raise IOError(f"base file for segmentation not defined")
        if os.path.isabs(base_file):
            file_path = base_file
        else:
            if not isinstance(load_locations[0], str):
                raise IOError(f"Cannot use relative path {base_file} for non path argument")
            file_path = os.path.join(os.path.dirname(load_locations[0]), base_file)
        if not os.path.exists(file_path):
            raise IOError(f"Base file for segmentation do not exists: {base_file} -> {file_path}")
        if metadata is None:
            metadata = {"default_spacing": [1, 1, 1]}
        image = ImageReader.read_image(
            file_path, callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
        return seg._replace(file_path=image.file_path, image=image)


class LoadImage(LoadBase):
    @classmethod
    def get_name(cls):
        return "Image(*.tif *.tiff *.lsm)"

    @classmethod
    def get_short_name(cls):
        return "img"

    @classmethod
    def load(cls, load_locations: typing.List[typing.Union[str, BytesIO, Path]],
             range_changed: typing.Callable[[int, int], typing.Any] = None,
             step_changed: typing.Callable[[int], typing.Any] = None, metadata: typing.Optional[dict] = None):
        if metadata is None:
            metadata = {"default_spacing": [1, 1, 1]}
        image = ImageReader.read_image(
            load_locations[0], callback_function=partial(proxy_callback, range_changed, step_changed),
            default_spacing=metadata["default_spacing"])
        return SegmentationTuple(image.file_path, image, None, [])


class SaveSegmentation(SaveBase):
    @classmethod
    def get_name(cls):
        return "Segmentation (*.seg *.tgz)"

    @classmethod
    def get_short_name(cls):
        return "seg"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("relative_path", "Relative Path\nin segmentation", False)]

    @classmethod
    def save(cls, save_location: typing.Union[str, BytesIO, Path], project_info: SegmentationTuple, parameters: dict,
             range_changed=None, step_changed=None):
        save_stack_segmentation(save_location, project_info, parameters)


def save_components(image: Image, components: list, segmentation: np.ndarray, dir_path: str,
                    range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun

    file_name = os.path.splitext(os.path.basename(image.file_path))[0]
    range_changed(0, 2 * len(components))
    for i in components:
        im = image.cut_image(segmentation == i, replace_mask=True)
        # print(f"[run] {im}")
        ImageWriter.save(im, os.path.join(dir_path, f"{file_name}_component{i}.tif"))
        step_changed(2 * i + 1)
        ImageWriter.save_mask(im, os.path.join(dir_path, f"{file_name}_component{i}_mask.tif"))
        step_changed(2 * i + 2)

# TODO create SaveComponents class
# class SaveComponents(SaveBase):


load_dict = Register(LoadImage, LoadSegmentationImage)
