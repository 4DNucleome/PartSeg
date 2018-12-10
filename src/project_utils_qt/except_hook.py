import sys
import partseg_utils.report_utils as report_utils


def my_excepthook(type_, value, trace_back):
    # log the exception here
    if report_utils.report_errors:
        try:
            # noinspection PyUnresolvedReferences
            from PyQt5.QtWidgets import QApplication
            if QApplication.instance():
                # noinspection PyUnresolvedReferences
                from project_utils_qt.error_dialog import ErrorDialog
                dial = ErrorDialog(value, "Exception during program run")
                dial.exec()
        except ImportError:
            pass
    # then call the default handler
    sys.__excepthook__(type_, value, trace_back)
