import os
import tarfile
import numpy as np
import json
from typing import Union, Optional
from io import BytesIO, TextIOBase, BufferedIOBase, RawIOBase, IOBase
from partseg_utils.io_utils import get_tarinfo
from tiff_image import Image, ImageWriter


def save_stack_segmentation(file: Union[tarfile.TarFile, str, TextIOBase, BufferedIOBase, RawIOBase, IOBase],
                            segmentation: np.ndarray, list_of_components, base_file: Optional[str]=None,
                            range_changed=None, step_changed=None):
    if range_changed is None:
        range_changed = empty_fun
    if step_changed is None:
        step_changed = empty_fun
    range_changed(0,5)
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
    np.save(segmentation_buff, segmentation)
    segmentation_tar = get_tarinfo("segmentation.npy", segmentation_buff)
    tar_file.addfile(segmentation_tar, fileobj=segmentation_buff)
    step_changed(3)
    metadata = {"components": list_of_components, "shape": segmentation.shape}
    if base_file is not None:
        metadata["base_file"] = base_file
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
