import os
from typing import Any, Optional

import numpy
from napari_plugin_engine import napari_hook_implementation

from PartSegCore.mask.io_functions import MaskProjectTuple, SaveROI


@napari_hook_implementation
def napari_write_labels(path: str, data: Any, meta: dict) -> Optional[str]:
    if not isinstance(data, numpy.ndarray):
        return
    ext = os.path.splitext(path)[1]
    if ext in SaveROI.get_extensions():
        project = MaskProjectTuple(file_path="", image=None, roi=data)
        SaveROI.save(path, project, parameters={})
        return path
