import os
import tarfile
import typing
from pathlib import Path

import numpy as np
import json
from typing import Union
from io import BytesIO, TextIOBase, BufferedIOBase, RawIOBase, IOBase
from ..partseg_utils.io_utils import get_tarinfo, SaveBase
from ..partseg_utils.segmentation.algorithm_describe_base import AlgorithmProperty
from PartSeg.tiff_image import Image, ImageWriter


class SegmentationTuple(typing.NamedTuple):
    image: Image
    segmentation: np.ndarray
    list_of_components: typing.List[int]


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
    metadata = {"components": segmentation_info.list_of_components, "shape": segmentation_info.segmentation.shape}
    if segmentation_info.image.file_path != "":
        if parameters["relative_path"] and isinstance(file, str):
            metadata["base_file"] = os.path.relpath(segmentation_info.image.file_path, os.path.dirname(file))
        else:
            metadata["base_file"] = segmentation_info.image.file_path
    metadata_buff = BytesIO(json.dumps(metadata).encode('utf-8'))
    metadata_tar = get_tarinfo("metadata.json", metadata_buff)
    tar_file.addfile(metadata_tar, metadata_buff)
    step_changed(4)
    if isinstance(file, str):
        tar_file.close()
    step_changed(5)


def load_stack_segmentation(file, range_changed=None, step_changed=None):
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
    metadata = json.loads(tar_file.extractfile("metadata.json").read().decode("utf8"))
    step_changed(4)
    if isinstance(file, str):
        tar_file.close()
    return segmentation, metadata


def empty_fun(_a0=None, _a1=None):
    pass


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
    def save(cls, save_location: typing.Union[str, BytesIO, Path], segmentation_info: SegmentationTuple,
             parameters: dict, range_changed=None, step_changed=None):
        save_stack_segmentation(save_location, segmentation_info, parameters,
                                range_changed=range_changed, step_changed=step_changed)


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
