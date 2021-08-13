import os
from typing import Any, Optional

import numpy
from napari_plugin_engine import napari_hook_implementation

from PartSegImage import Image, ImageWriter
from PartSegImage.image import DEFAULT_SCALE_FACTOR


@napari_hook_implementation
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    if not isinstance(data, numpy.ndarray):
        return
    ext = os.path.splitext(path)[1]
    if ext in {".tiff", ".tif", ".TIFF", ".TIF"}:
        scale_shift = min(data.ndim, 3)
        image = Image(
            data,
            numpy.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
            axes_order="TZXY"[-data.ndim :],
            channel_names=[meta["name"]],
        )
        ImageWriter.save(image, path)
        return path


@napari_hook_implementation
def napari_write_image(path: str, data: Any, meta: dict) -> Optional[str]:
    if not isinstance(data, numpy.ndarray):
        return
    ext = os.path.splitext(path)[1]
    if ext in {".tiff", ".tif", ".TIFF", ".TIF"}:
        scale_shift = min(data.ndim, 3)
        axes = "TZXY"
        channel_names = [meta["name"]]
        if data.shape[-1] < 6:
            axes += "C"
            scale_shift -= 1
            channel_names = None
        image = Image(
            data,
            numpy.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
            axes_order=axes[-data.ndim :],
            channel_names=channel_names,
        )
        ImageWriter.save(image, path)
        return path
