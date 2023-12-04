import os

from napari.layers import Layer

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg.common_backend import napari_get_settings
from PartSegCore.json_hooks import PartSegEncoder

_SETTINGS = None


class PartSegNapariEncoder(PartSegEncoder):
    def default(self, o):
        if isinstance(o, Layer):
            return o.name
        return super().default(o)


class PartSegNapariSettings(PartSettings):
    json_encoder_class = PartSegNapariEncoder


def get_settings() -> PartSettings:
    global _SETTINGS  # noqa: PLW0603  # pylint: disable=global-statement
    if _SETTINGS is None:
        napari_settings = napari_get_settings()
        if hasattr(napari_settings, "path"):
            save_path = napari_settings.path
        else:
            save_path = os.path.dirname(napari_settings.config_path)
        _SETTINGS = PartSegNapariSettings(os.path.join(save_path, "PartSeg_napari_plugins"))
        _SETTINGS.load()
    return _SETTINGS
