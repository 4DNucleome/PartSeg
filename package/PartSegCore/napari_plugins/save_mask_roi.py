import os
from typing import Any, Optional

import numpy as np
from napari_plugin_engine import napari_hook_implementation

from PartSegCore.mask.io_functions import MaskProjectTuple, SaveROI
from PartSegCore.roi_info import ROIInfo
from PartSegImage.image import DEFAULT_SCALE_FACTOR


@napari_hook_implementation
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    if not isinstance(data, np.ndarray):
        return None
    ext = os.path.splitext(path)[1]
    if ext in SaveROI.get_extensions():
        project = MaskProjectTuple(file_path="", image=None, roi_info=ROIInfo(data))
        SaveROI.save(path, project, parameters={"spacing": np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-3:]})
        return path
