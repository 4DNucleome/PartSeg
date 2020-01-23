import sys
from PartSegCore import state_store


def my_excepthook(type_, value, trace_back):
    """
    Custom excepthook. base on base on :py:data:`state_store.show_error_dialog` decide if shown error dialog.

    """

    # log the exception here
    if state_store.show_error_dialog:
        try:
            # noinspection PyUnresolvedReferences
            from qtpy.QtWidgets import QApplication

            if QApplication.instance():
                from qtpy.QtCore import Qt, QThread
                from qtpy.QtCore import QMetaObject

                QApplication.instance().error = value
                if QThread.currentThread() != QApplication.instance().thread():
                    QMetaObject.invokeMethod(QApplication.instance(), "show_error", Qt.QueuedConnection)
                else:
                    QApplication.instance().show_error()
        except ImportError:
            sys.__excepthook__(type_, value, trace_back)
    else:
        # then call the default handler
        sys.__excepthook__(type_, value, trace_back)
