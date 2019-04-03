import sys
from qtpy.QtCore import Slot, QThread
from qtpy.QtWidgets import QApplication, QMessageBox
import packaging.version
from xmlrpc import client

from .. import __version__
from ..utils import state_store


class CheckVersionThread(QThread):
    def __init__(self):
        super().__init__()
        self.release = __version__

    def run(self):
        try:
            proxy = client.ServerProxy('http://pypi.python.org/pypi')
            self.release = proxy.package_releases("PartSeg")[0]
        except:
            pass


class CustomApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.error = None
        self.warning = None, None
        self.release_check = CheckVersionThread()
        self.release_check.finished.connect(self._check_release)

    @Slot()
    def show_error(self):
        if self.error is None:
            return
        from ..project_utils_qt.error_dialog import ErrorDialog
        dial = ErrorDialog(self.error, "Exception during program run")
        # TODO check
        # dial.moveToThread(QApplication.instance().thread())
        dial.exec()

    @Slot()
    def show_warning(self):
        if not isinstance(self.warning, (list, tuple)) or self.warning[0] is None:
            return
        message = QMessageBox(QMessageBox.Warning, self.warning[0], self.warning[1], QMessageBox.Ok)
        message.exec()

    def check_release(self):
        if state_store.check_for_updates:
            self.release_check.start()

    def _check_release(self):
        my_version = packaging.version.parse(__version__)
        remote_version = packaging.version.parse(self.release_check.release)
        if remote_version > my_version:
            if getattr(sys, 'frozen', False):
                message = QMessageBox(
                    QMessageBox.Information, "New release",
                    f"You use outdated version of PartSeg. Your version is {my_version} and current is {remote_version}. "
                    "You can download next release form https://4dnucleome.cent.uw.edu.pl/PartSeg/", QMessageBox.Ok)
            else:
                message = QMessageBox(
                    QMessageBox.Information, "New release",
                    f"You use outdated version of PartSeg. Your version is {my_version} and current is {remote_version}. "
                    "You can update it from pypi (pip install -U PartSeg)", QMessageBox.Ok)

            message.exec()





