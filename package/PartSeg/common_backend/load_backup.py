import os
import shutil
from glob import glob

import packaging.version
from qtpy.QtWidgets import QMessageBox, QWidget

from PartSegCore import state_store

from .. import parsed_version
from .._launcher.check_version import IGNORE_FILE
from . import napari_get_settings


def import_config():
    if os.path.exists(state_store.save_folder):
        return
    version = packaging.version.parse(parsed_version.base_version)
    base_folder = os.path.dirname(state_store.save_folder)
    possible_folders = glob(os.path.join(base_folder, "*"))
    versions = list(
        sorted(
            (
                x
                for x in [packaging.version.parse(os.path.basename(y)) for y in possible_folders]
                if isinstance(x, packaging.version.Version)
            ),
            reverse=True,
        )
    )
    before_version = None
    for x in versions:
        if x < version:
            before_version = x
            break
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
            elif hasattr(napari_settings, "_load") and napari_settings._load is not None:  # pylint: disable=W0212
                napari_settings._load()  # pylint: disable=W0212
