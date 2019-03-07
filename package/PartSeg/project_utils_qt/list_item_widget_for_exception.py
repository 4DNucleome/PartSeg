import typing
from qtpy.QtWidgets import QListWidgetItem, QListWidget

from PartSeg.project_utils_qt.error_dialog import ErrorDialog


class ExceptionListItem(QListWidgetItem):
    def __init__(self, exception: typing.Union[Exception, typing.Tuple[Exception, typing.List]],
                 parent: QListWidget = None):
        if isinstance(exception, Exception):
            super().__init__(f"{type(exception)}: {exception}", parent, QListWidgetItem.UserType)
            self.exception = exception
            self.traceback_summary = None
        else:
            super().__init__(f"{type(exception[0])}: {exception[0]}", parent, QListWidgetItem.UserType)
            self.exception = exception[0]
            self.traceback_summary = exception[1]

        self.setToolTip("Double click for report")


class ExceptionList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.itemDoubleClicked.connect(self.item_double_clicked)

    @staticmethod
    def item_double_clicked(el: QListWidgetItem):
        if isinstance(el, ExceptionListItem):
            dial = ErrorDialog(el.exception, "Error during batch processing", traceback_summary=el.traceback_summary)
            dial.exec()

