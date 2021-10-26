"""
This module contains non gui Qt based components
"""
import os.path
from typing import TYPE_CHECKING

import napari
import packaging.version

if TYPE_CHECKING:  # pragma: no cover
    from napari.settings import NapariSettings

try:
    from napari.settings import get_settings as _napari_get_settings
except ImportError:  # pragma: no cover
    try:
        from napari.utils.settings import get_settings as _napari_get_settings
    except ImportError:
        from napari.utils.settings import SETTINGS

        def _napari_get_settings(path=None):
            return SETTINGS


def napari_get_settings(path=None) -> "NapariSettings":
    if path is not None and packaging.version.parse(napari.__version__) >= packaging.version.parse("0.4.11"):
        path = os.path.join(path, "settings.yaml")

    try:
        return _napari_get_settings(path)
    except:  # noqa  # pylint: disable=W0702
        return _napari_get_settings()


__all__ = ("napari_get_settings",)
