import sys

import sentry_sdk
from qtpy.QtWidgets import QMessageBox
from superqt import ensure_main_thread

from PartSeg import parsed_version
from PartSegCore import state_store
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegImage import TiffFileException


def my_excepthook(type_, value, trace_back):
    """
    Custom excepthook. base on base on :py:data:`state_store.show_error_dialog` decide if shown error dialog.

    """

    # log the exception here
    if state_store.show_error_dialog and not isinstance(value, KeyboardInterrupt):
        if state_store.report_errors and parsed_version.is_devrelease:
            with sentry_sdk.push_scope() as scope:
                scope.set_tag("auto_report", "true")
                sentry_sdk.capture_exception(value)
        try:
            show_error(value)
        except ImportError:
            sys.__excepthook__(type_, value, trace_back)
    elif isinstance(value, KeyboardInterrupt):
        print("KeyboardInterrupt close", file=sys.stderr)
        sys.exit(1)
    else:
        # then call the default handler
        sys.__excepthook__(type_, value, trace_back)


@ensure_main_thread
def show_error(error=None):
    """This class create error dialog and show it"""
    if error is None:
        return

    if isinstance(error, TiffFileException):
        mess = QMessageBox()
        mess.setIcon(QMessageBox.Critical)
        mess.setText("During read file there is an error: " + error.args[0])
        mess.setWindowTitle("Tiff error")
        mess.exec_()
        return
    if isinstance(error, SegmentationLimitException):
        mess = QMessageBox()
        mess.setIcon(QMessageBox.Critical)
        mess.setText("During segmentation process algorithm meet limitations:\n" + "\n".join(error.args))
        mess.setWindowTitle("Segmentation limitations")
        mess.exec_()
        return
    from PartSeg.common_gui.error_report import ErrorDialog

    dial = ErrorDialog(error, "Exception during program run")
    dial.exec_()


@ensure_main_thread
def show_warning(header=None, text=None):
    """show warning :py:class:`PyQt5.QtWidgets.QMessageBox`"""
    message = QMessageBox(QMessageBox.Warning, header, text, QMessageBox.Ok)
    message.exec_()
