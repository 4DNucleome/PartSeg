import os
import shutil
from glob import glob

import packaging.version
from qtpy.QtWidgets import QMessageBox, QWidget

from PartSegCore import state_store

from .. import __version__


def import_config():
    if os.path.exists(state_store.save_folder):
        return
    version = packaging.version.parse(packaging.version.parse(__version__).base_version)
    base_folder = os.path.dirname(state_store.save_folder)
    possible_folders = glob(os.path.join(base_folder, "*"))
    versions = list(
        sorted(
            [
                x
                for x in [packaging.version.parse(os.path.basename(y)) for y in possible_folders]
                if isinstance(x, packaging.version.Version)
            ],
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
