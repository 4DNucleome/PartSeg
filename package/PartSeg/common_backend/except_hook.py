import logging
import sys

import sentry_sdk
from qtpy.QtCore import QCoreApplication, QThread
from qtpy.QtWidgets import QMessageBox
from superqt import ensure_main_thread

from PartSeg import state_store
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegImage import TiffFileException


def my_excepthook(type_, value, trace_back):
    """
    Custom excepthook.
    Close application on :py:class:`KeyboardInterrupt`.

    If :py:data:`PartSeg.state_store.always_report` is set then just sent report using sentry.
    otherwise show dialog with information about error and ask user
    if he wants to send report using :py:func:`show_error`.
    """

    # log the exception here
    if state_store.show_error_dialog and not isinstance(value, KeyboardInterrupt):
        if state_store.auto_report or state_store.always_report:
            with sentry_sdk.new_scope() as scope:
                scope.set_tag("auto_report", "true")
                scope.set_tag("main_thread", QCoreApplication.instance().thread() == QThread.currentThread())
                sentry_sdk.capture_exception(value)
        if state_store.always_report:
            return
        try:
            show_error(value)
        except ImportError:
            sys.__excepthook__(type_, value, trace_back)
    elif isinstance(value, KeyboardInterrupt):
        logging.warning("KeyboardInterrupt close")
        sys.exit(1)
    else:
        # then call the default handler
        sys.__excepthook__(type_, value, trace_back)


@ensure_main_thread
def show_error(error=None):
    """
    For :py:class:`SegmentationLimitException` and :py:class:`TiffFileException`
    show dialog with information about problem.

    For other exceptions show :py:class:`ErrorDialog` dialog
    with information about error that allow to report it.

    :param error: exception to show
    """
    if error is None:
        return

    from PartSeg.common_gui.error_report import ErrorDialog, QMessageFromException

    if isinstance(error, TiffFileException):
        QMessageFromException.critical(
            None,
            "Tiff file error",
            f"During read file there is an error: {', '.join(str(x) for x in error.args)}",
            exception=error,
        )
        return
    if isinstance(error, SegmentationLimitException):
        QMessageFromException.critical(
            None,
            "Segmentation limitations",
            f"During segmentation process algorithm meet limitations: {', '.join(str(x) for x in error.args)}",
            exception=error,
        )
        return

    dial = ErrorDialog(error, "Exception during program run")
    dial.exec_()


@ensure_main_thread
def show_warning(header=None, text=None, exception=None):
    """
    Show warning :py:class:`PyQt5.QtWidgets.QMessageBox`

    This function is to ensure creation warning dialog in main thread.
    """
    if exception is not None:
        from PartSeg.common_gui.error_report import QMessageFromException

        message = QMessageFromException(QMessageBox.Icon.Warning, header, text, exception=exception)
    else:
        message = QMessageBox(QMessageBox.Icon.Warning, header, text, QMessageBox.StandardButton.Ok)
    message.exec_()
