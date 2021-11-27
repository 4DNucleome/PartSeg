import os

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg.common_backend import napari_get_settings

# from PartSeg.common_backend.base_settings import BaseSettings

_settings = None


def get_settings() -> PartSettings:
    global _settings  # pylint: disable=W0603
    if _settings is None:
        napari_settings = napari_get_settings()
        if hasattr(napari_settings, "path"):
            save_path = napari_settings.path
        else:
            save_path = os.path.dirname(napari_settings.config_path)
        _settings = PartSettings(os.path.join(save_path, "PartSeg_napari_plugins"))
        _settings.load()
    return _settings
