"""
This module contains non gui Qt based components
"""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.settings import NapariSettings

try:
    from napari.settings import get_settings as _napari_get_settings
except ImportError:
    try:
        from napari.utils.settings import get_settings as _napari_get_settings
    except ImportError:
        from napari.utils.settings import SETTINGS

        def _napari_get_settings(path=None):
            return SETTINGS


def napari_get_settings(path=None) -> "NapariSettings":
    try:
        return _napari_get_settings(path)
    except:  # noqa  # pylint: disable=W0702
        return _napari_get_settings()


__all__ = ("napari_get_settings",)
