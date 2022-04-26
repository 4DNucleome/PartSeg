"""
This module contains non gui Qt based components
"""
import os.path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from napari.settings import NapariSettings

from napari.settings import get_settings as _napari_get_settings


def napari_get_settings(path=None) -> "NapariSettings":
    if path is not None:
        path = os.path.join(path, "settings.yaml")

    try:
        return _napari_get_settings(path)
    except:  # noqa  # pylint: disable=W0702
        return _napari_get_settings()


__all__ = ("napari_get_settings",)
