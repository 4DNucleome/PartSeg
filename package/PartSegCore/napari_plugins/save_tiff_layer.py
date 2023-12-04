import os
from typing import Any, List, Optional

import numpy as np
from napari.types import FullLayerData

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


def napari_write_images(path: str, layer_data: List[FullLayerData]) -> List[str]:
    ext = os.path.splitext(path)[1]
    base_shape = layer_data[0][0].shape
    if not all(isinstance(x[0], np.ndarray) and x[0].shape == base_shape for x in layer_data) or ext.lower() not in {
        ".tiff",
        ".tif",
    }:
        return []
    scale_shift = min(len(base_shape), 3)
    axes = "TZXY"
    channel_names = [meta["name"] for _, meta, _ in layer_data]
    meta = layer_data[0][1]
    if len(layer_data) == 1:
        data = layer_data[0][0]
        if data.shape[-1] < 6:
            axes += "C"
            scale_shift -= 1
            channel_names = [f'{meta["name"]} {i}' for i in range(1, data.shape[-1] + 1)]
        axes = axes[-data.ndim :]
    else:
        data = [x[0] for x in layer_data]
        axes = f"C{axes[-len(data[0].shape):]}"
        scale_shift -= 1
    image = Image(
        data,
        np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        axes_order=axes,
        channel_names=channel_names,
        shift=np.divide(meta["translate"], DEFAULT_SCALE_FACTOR)[-scale_shift:],
        name="Image",
    )
    ImageWriter.save(image, path)
    return [path]
