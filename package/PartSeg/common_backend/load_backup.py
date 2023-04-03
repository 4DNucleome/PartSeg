import os
import shutil
from glob import glob

import packaging.version
from qtpy.QtWidgets import QMessageBox, QWidget

from PartSeg import parsed_version, state_store
from PartSeg._launcher.check_version import IGNORE_FILE
from PartSeg.common_backend import napari_get_settings


def _parse_version(name):
    try:
        return packaging.version.parse(name)
    except packaging.version.InvalidVersion:
        return None


def import_config():
    """
    Check if settings folder for current version already exists.

    Otherwise, when settings for previous PartSeg version exists,
    ask user if he would to import settings from the newest one.
    """
    if os.path.exists(state_store.save_folder):
        return
    version = packaging.version.parse(parsed_version.base_version)
    base_folder = os.path.dirname(state_store.save_folder)
    possible_folders = glob(os.path.join(base_folder, "*"))
    versions = sorted(
        (
            x
            for x in [_parse_version(os.path.basename(y)) for y in possible_folders]
            if isinstance(x, packaging.version.Version)
        ),
        reverse=True,
    )
    before_version = next((x for x in versions if x < version), None)
    if before_version is not None:
        before_name = str(before_version)
        widget = QWidget()
        resp = QMessageBox.question(
            widget,
            "Import from old version",
            "There is no configuration folder for this version of PartSeg\n"
            "Would you like to import it from " + before_name + " version of PartSeg",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            shutil.copytree(os.path.join(base_folder, before_name), state_store.save_folder)
            if os.path.exists(os.path.join(state_store.save_folder, IGNORE_FILE)):
                os.remove(os.path.join(state_store.save_folder, IGNORE_FILE))
            napari_settings = napari_get_settings(state_store.save_folder)
            if hasattr(napari_settings, "load") and napari_settings.load is not None:
                napari_settings.load()
            elif getattr(napari_settings, "_load", None) is not None:
                napari_settings._load()  # pylint: disable=protected-access
