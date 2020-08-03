import sys

import sentry_sdk

from PartSeg import parsed_version
from PartSegCore import state_store


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
            # noinspection PyUnresolvedReferences
            from qtpy.QtWidgets import QApplication

            if QApplication.instance():
                from qtpy.QtCore import QMetaObject, Qt, QThread

                QApplication.instance().error = value
                if QThread.currentThread() != QApplication.instance().thread():
                    QMetaObject.invokeMethod(QApplication.instance(), "show_error", Qt.QueuedConnection)
                else:
                    QApplication.instance().show_error()
        except ImportError:
            sys.__excepthook__(type_, value, trace_back)
    elif isinstance(value, KeyboardInterrupt):
        print("KeyboardInterrupt close", file=sys.stderr)
        sys.exit(1)
    else:
        # then call the default handler
        sys.__excepthook__(type_, value, trace_back)
