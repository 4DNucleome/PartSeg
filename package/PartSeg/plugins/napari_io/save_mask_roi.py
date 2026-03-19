import os
from typing import Any, Optional

import numpy as np

from PartSegCore.mask.io_functions import MaskProjectTuple, SaveROI, SaveROIOptions
from PartSegCore.roi_info import ROIInfo
from PartSegImage.image import DEFAULT_SCALE_FACTOR


def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    if not isinstance(data, np.ndarray):
        return None
    ext = os.path.splitext(path)[1]
    if ext in SaveROI.get_extensions():
        project = MaskProjectTuple(file_path="", image=None, roi_info=ROIInfo(data))
        SaveROI.save(
            path, project, parameters=SaveROIOptions(spacing=list(np.divide(meta["scale"], DEFAULT_SCALE_FACTOR)[-3:]))
        )
        return path
    return None
