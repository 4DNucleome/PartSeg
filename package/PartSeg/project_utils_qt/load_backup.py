import os
import packaging.version
import shutil
from glob import glob

from PyQt5.QtWidgets import QMessageBox, QWidget

from .. import CONFIG_FOLDER, __version__

def import_config():
    if not os.path.exists(CONFIG_FOLDER):
        version = packaging.version.parse(__version__)
        base_folder = os.path.dirname(CONFIG_FOLDER)
        possible_folders = glob(os.path.join(base_folder, "*"))
        versions = list(
            sorted([x for x in
                    [packaging.version.parse(os.path.basename(y)) for y in possible_folders]
                    if isinstance(x, packaging.version.Version)],
                   reverse=True))
        before_version = None
        for x in versions:
            if x < version:
                before_version = x
                break
        if before_version is not None:
            before_name = str(before_version)
            widget = QWidget()
            resp = QMessageBox.question(widget, "Import from old version",
                                        "There is no configuration folder for this version of PartSeg\n"
                                        "Would you like to import it from " + before_name + " version of PartSeg",
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if resp == QMessageBox.Yes:
                shutil.copytree(os.path.join(base_folder, before_name), os.path.join(base_folder, __version__))
