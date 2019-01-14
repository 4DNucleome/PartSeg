from qtpy.QtCore import Slot
from qtpy.QtWidgets import QApplication

class CustomApplication(QApplication):
    def __init__(self, argv):
        super().__init__(argv)
        self.value = None

    @Slot()
    def show_error(self):
        if self.value is None:
            return
        from ..project_utils_qt.error_dialog import ErrorDialog
        dial = ErrorDialog(self.value, "Exception during program run")
        dial.moveToThread(QApplication.instance().thread())
        dial.exec()