import sys
from qtpy.QtCore import Slot, QThread
from qtpy.QtWidgets import QApplication, QMessageBox
import packaging.version
from xmlrpc import client

from PartSegImage import TiffFileException
from .. import __version__
from ..utils import state_store


class CheckVersionThread(QThread):
    """Therad to check if there is new PartSeg release. Checks base on newest version available on pypi_

     .. _PYPI: https://pypi.org/project/PartSeg/
     """
    def __init__(self):
        super().__init__()
        self.release = __version__

    def run(self):
        """This function perform check"""

        # noinspection PyBroadException
        try:
            proxy = client.ServerProxy('http://pypi.python.org/pypi')
            self.release = proxy.package_releases("PartSeg")[0]
        except:
            pass


class CustomApplication(QApplication):
    """
    This class is created because Qt do not allows to create GUI elements outside main thread.
    usage can bee seen in :py:func:`PartSeg.common_backend.except_hook.my_excepthook`

    :ivar error: :py:class:`Exception` to be show in error dialog
    :ivar warning: Pair of strings. First is set as title, second as content of :py:class:`PyQt5.QtWidgets.QMessageBox`
    """
    def __init__(self, argv):
        super().__init__(argv)
        self.error = None
        self.warning = "", ""
        self.release_check = CheckVersionThread()
        self.release_check.finished.connect(self._check_release)

    @Slot()
    def show_error(self):
        """This class create error dialog and show it"""
        if self.error is None:
            return
        from PartSeg.common_gui.error_report import ErrorDialog
        if isinstance(self.error, TiffFileException):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("During read file there is an error: " + self.error.args[0])
            mess.setWindowTitle("Tiff error")
            mess.exec()
            return
        dial = ErrorDialog(self.error, "Exception during program run")
        # TODO check
        # dial.moveToThread(QApplication.instance().thread())
        dial.exec()

    @Slot()
    def show_warning(self):
        """show warning :py:class:`PyQt5.QtWidgets.QMessageBox`"""
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
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    "You can download next release form https://4dnucleome.cent.uw.edu.pl/PartSeg/", QMessageBox.Ok)
            else:
                message = QMessageBox(
                    QMessageBox.Information, "New release",
                    f"You use outdated version of PartSeg. "
                    f"Your version is {my_version} and current is {remote_version}. "
                    "You can update it from pypi (pip install -U PartSeg)", QMessageBox.Ok)

            message.exec()
