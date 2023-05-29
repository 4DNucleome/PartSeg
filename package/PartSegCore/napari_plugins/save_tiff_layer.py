import os
from typing import Any, Optional

import numpy as np

from PartSegImage import Image, ImageWriter
from PartSegImage.image import DEFAULT_SCALE_FACTOR


def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    ext = os.path.splitext(path)[1]
    if not isinstance(data, np.ndarray) or ext not in {".tiff", ".tif", ".TIFF", ".TIF"}:
        return None
    scale_shift = min(data.ndim, 3)
    image = Image(
        data,
        np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        axes_order="TZYX"[-data.ndim :],
        channel_names=[meta["name"]],
        shift=np.divide(meta["translate"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        name="ROI",
    )
    ImageWriter.save(image, path)
    return path


def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    ext = os.path.splitext(path)[1]
    if not isinstance(data, np.ndarray) or ext not in {".tiff", ".tif", ".TIFF", ".TIF"}:
        return None
    scale_shift = min(data.ndim, 3)
    axes = "TZXY"
    channel_names = [meta["name"]]
    if data.shape[-1] < 6:
        axes += "C"
        scale_shift -= 1
        channel_names = [f'{meta["name"]} {i}' for i in range(1, data.shape[-1] + 1)]
    image = Image(
        data,
        np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        axes_order=axes[-data.ndim :],
        channel_names=channel_names,
        shift=np.divide(meta["translate"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        name="Image",
    )
    ImageWriter.save(image, path)
    return path
