import sys
from ..partseg_utils import report_utils


def my_excepthook(type_, value, trace_back):
    # log the exception here
    if report_utils.report_errors:
        try:
            # noinspection PyUnresolvedReferences
            from qtpy.QtWidgets import QApplication
            if QApplication.instance():
                from qtpy.QtCore import Qt
                from qtpy.QtCore import QMetaObject
                QApplication.instance().value = value
                QMetaObject.invokeMethod(QApplication.instance(), "show_error",  Qt.QueuedConnection)
        except ImportError:
            pass
    # then call the default handler
    sys.__excepthook__(type_, value, trace_back)
