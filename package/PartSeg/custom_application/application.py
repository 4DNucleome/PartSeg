import json
import sys
import urllib.error
import urllib.request

import packaging.version
import sentry_sdk
from qtpy.QtCore import QThread
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QApplication, QMessageBox
from superqt import ensure_main_thread

from PartSegCore import state_store
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegImage import TiffFileException

from .. import __version__


class CheckVersionThread(QThread):
    """Thread to check if there is new PartSeg release. Checks base on newest version available on pypi_

    .. _PYPI: https://pypi.org/project/PartSeg/
    """

    def __init__(self):
        super().__init__()
        self.release = __version__
        self.url = "https://4dnucleome.cent.uw.edu.pl/PartSeg/"

    def run(self):
        """This function perform check"""

        # noinspection PyBroadException
        try:
            r = urllib.request.urlopen("https://pypi.org/pypi/PartSeg/json")  # nosec
            data = json.load(r)
            self.release = data["info"]["version"]
            self.url = data["info"]["home_page"]
        except (KeyError, urllib.error.URLError):
            pass
        except Exception as e:  # pylint: disable=W0703
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("auto_report", "true")
                scope.set_tag("check_version", "true")
                sentry_sdk.capture_exception(e)


class CustomApplication(QApplication):
    """
    This class is created because Qt do not allows to create GUI elements outside main thread.
    usage can bee seen in :py:func:`PartSeg.common_backend.except_hook.my_excepthook`

    :ivar error: :py:class:`Exception` to be show in error dialog
    :ivar warning: Pair of strings. First is set as title, second as content of :py:class:`PyQt5.QtWidgets.QMessageBox`
    """

    def __init__(self, argv, name, icon):
        super().__init__(argv)
        self.error = None
        self.warning = "", ""
        self.release_check = CheckVersionThread()
        self.release_check.finished.connect(self._check_release)
        self.setWindowIcon(QIcon(icon))
        self.setApplicationName(name)

    @ensure_main_thread
    def show_error(self, error=None):
        """This class create error dialog and show it"""
        if error is None:
            error = self.error
        if error is None:
            return
        from PartSeg.common_gui.error_report import ErrorDialog

        if isinstance(error, TiffFileException):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("During read file there is an error: " + self.error.args[0])
            mess.setWindowTitle("Tiff error")
            mess.exec()
            return
        if isinstance(error, SegmentationLimitException):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("During segmentation process algorithm meet limitations:\n" + "\n".join(self.error.args))
            mess.setWindowTitle("Segmentation limitations")
            mess.exec()
            return
        dial = ErrorDialog(error, "Exception during program run")
        # TODO check
        # dial.moveToThread(QApplication.instance().thread())
        dial.exec()

    @ensure_main_thread
    def show_warning(self, header=None, text=None):
        """show warning :py:class:`PyQt5.QtWidgets.QMessageBox`"""
        if text is None:
            if isinstance(self.warning, (list, tuple)) and len(self.warning) == 2:
                return
            header, text = self.warning
        message = QMessageBox(QMessageBox.Warning, header, text, QMessageBox.Ok)
        message.exec()

    def check_release(self):
        if state_store.check_for_updates:
            self.release_check.start()

    def _check_release(self):
        my_version = packaging.version.parse(__version__)
        remote_version = packaging.version.parse(self.release_check.release)
        if remote_version > my_version:
            if getattr(sys, "frozen", False):
                message = QMessageBox(
                    QMessageBox.Information,
                    "New release",
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    f"You can download next release form {self.release_check.url}",
                    QMessageBox.Ok,
                )
            else:
                message = QMessageBox(
                    QMessageBox.Information,
                    "New release",
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    "You can update it from pypi (pip install -U PartSeg)",
                    QMessageBox.Ok,
                )

            message.exec()
