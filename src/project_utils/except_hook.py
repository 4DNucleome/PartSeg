import sys
import sentry_sdk


sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")


def my_excepthook(type_, value, tback):
    # log the exception here
    try:
        # noinspection PyUnresolvedReferences
        from PyQt5.QtWidgets import QApplication
        if QApplication.instance():
            # noinspection PyUnresolvedReferences
            from .error_dialog import ErrorDialog
            dial = ErrorDialog(value, "Exception during program run")
            dial.exec()
    except ImportError:
        pass
    # then call the default handler
    sys.__excepthook__(type_, value, tback)
