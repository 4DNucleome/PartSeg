from .partseg_utils import HistoryElement, PartEncoder
import numpy as np
from tiff_image import Image, ImageWriter
import tarfile
from io import BytesIO, StringIO
import typing
import os.path
import json
import datetime

def get_tarinfo(name, buffer: typing.Union[BytesIO, StringIO]):
    tar_info = tarfile.TarInfo(name=name)
    buffer.seek(0)
    if isinstance(buffer, BytesIO):
        tar_info.size = len(buffer.getbuffer())
    else:
        tar_info.size = len(buffer.getvalue())
    tar_info.mtime = datetime.datetime.now().timestamp()
    return tar_info


def save_project(file_path: str, image: Image, segmentation: np.ndarray, full_segmentation: np.ndarray,
                 mask: typing.Union[np.ndarray, None], history: typing.List[HistoryElement], algorithm_parameters: dict):
    ext = os.path.splitext(file_path)[1]
    if ext.lower() in ['.bz2', ".tbz2"]:
        tar_mod = 'w:bz2'
    else:
        tar_mod = 'w:gz'
    with tarfile.open(file_path, tar_mod) as tar:
        sek_dkt = {"segmentation": segmentation, "full_segmentation": full_segmentation}
        if mask is not None:
            sek_dkt["mask"] = mask
        seg_buff = BytesIO()
        np.savez(seg_buff, sek_dkt)
        tar_numpy = get_tarinfo("segmentation.npz", seg_buff)
        tar.addfile(tarinfo=tar_numpy, fileobj=seg_buff)
        image_buff = BytesIO()
        ImageWriter.save(image, image_buff)
        tar_image = get_tarinfo("image.tif", image_buff)
        tar.addfile(tarinfo=tar_image, fileobj=image_buff)
        para_str = json.dumps(algorithm_parameters, cls=PartEncoder)
        parameters_buff = BytesIO(para_str.encode('utf-8'))
        tar_algorithm = get_tarinfo("algorithm.json", parameters_buff)
        tar.addfile(tar_algorithm, parameters_buff)
        el_info = []
        for i, el in enumerate(history):
            el_info.append({"index": i, "algorithm_name": el.algorithm_name, "values": el.algorithm_values,
                            "mask_property": el.mask_property})
            el.arrays.seek(0)
            hist_info = get_tarinfo(f"history/arrays_{i}.npz", el.arrays)
            el.arrays.seek(0)
            tar.addfile(hist_info, el.arrays)
        if len(el_info) > 0:
            hist_str = json.dumps(el_info, cls=PartEncoder)
            hist_buff = BytesIO(hist_str.encode('utf-8'))
            tar_algorithm = get_tarinfo("history/history.json", hist_buff)
            tar.addfile(tar_algorithm, hist_buff)
