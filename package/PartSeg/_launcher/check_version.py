import json
import os
import sys
import urllib.error
import urllib.request
from contextlib import suppress
from datetime import date

import packaging.version
import sentry_sdk
from qtpy.QtCore import QThread
from qtpy.QtWidgets import QMessageBox
from superqt import ensure_main_thread

from PartSegCore import state_store

from .. import __version__

IGNORE_DAYS = 21
IGNORE_FILE = "ignore.txt"


class CheckVersionThread(QThread):
    """Thread to check if there is new PartSeg release. Checks base on newest version available on pypi_

    .. _PYPI: https://pypi.org/project/PartSeg/
    """

    def __init__(
        self, package_name="PartSeg", default_url="https://4dnucleome.cent.uw.edu.pl/PartSeg/", base_version=__version__
    ):
        super().__init__()
        self.release = base_version
        self.base_release = base_version
        self.package_name = package_name
        self.url = default_url
        self.finished.connect(self.show_version_info)

    def run(self):
        """This function perform check"""

        # noinspection PyBroadException
        if not state_store.check_for_updates:
            return
        try:
            if os.path.exists(os.path.join(state_store.save_folder, IGNORE_FILE)):
                with open(os.path.join(state_store.save_folder, IGNORE_FILE)) as f_p, suppress(ValueError):
                    old_date = date.fromisoformat(f_p.read())
                    if (date.today() - old_date).days < IGNORE_DAYS:
                        return
                os.remove(os.path.join(state_store.save_folder, IGNORE_FILE))

            with urllib.request.urlopen(f"https://pypi.org/pypi/{self.package_name}/json") as r:  # nosec
                data = json.load(r)
            self.release = data["info"]["version"]
            self.url = data["info"]["home_page"]
        except (KeyError, urllib.error.URLError):  # pragma: no cover
            pass
        except Exception as e:  # pylint: disable=W0703
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("auto_report", "true")
                scope.set_tag("check_version", "true")
                sentry_sdk.capture_exception(e)

    @ensure_main_thread
    def show_version_info(self):
        my_version = packaging.version.parse(self.base_release)
        remote_version = packaging.version.parse(self.release)
        if remote_version > my_version:
            if getattr(sys, "frozen", False):
                message = QMessageBox(
                    QMessageBox.Information,
                    "New release",
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    f"You can download next release form {self.url}",
                    QMessageBox.Ok | QMessageBox.Ignore,
                )
            else:
                message = QMessageBox(
                    QMessageBox.Information,
                    "New release",
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    "You can update it from pypi (pip install -U PartSeg)",
                    QMessageBox.Ok | QMessageBox.Ignore,
                )

            if message.exec_() == QMessageBox.Ignore:
                with open(os.path.join(state_store.save_folder, IGNORE_FILE), "w") as f_p:
                    f_p.write(date.today().isoformat())
