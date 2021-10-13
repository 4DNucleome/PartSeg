import os

from PartSeg.common_backend import napari_get_settings
from PartSeg.common_backend.base_settings import BaseSettings

_settings = None


def get_settings() -> BaseSettings:
    global _settings  # pylint: disable=W0603
    if _settings is None:
        napari_settings = napari_get_settings()
        _settings = BaseSettings(os.path.join(napari_settings.path, "PartSeg_napari_plugins"))
        _settings.load()
    return _settings
