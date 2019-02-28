from qtpy.QtCore import Slot
from qtpy.QtWidgets import QApplication, QMessageBox


class CustomApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.error = None
        self.warning = None, None

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
