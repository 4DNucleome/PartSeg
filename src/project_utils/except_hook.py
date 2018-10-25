import sys
import traceback
import sentry_sdk
sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")

def my_excepthook(type_, value, tback):
    # log the exception here
    from PyQt5.QtWidgets import QMessageBox
    #TODO replace with custom dialog
    msg_box = QMessageBox()
    msg_box.setText("Some error occurs")
    msg_box.setInformativeText("Did you want to report it?")
    msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    msg_box.setDefaultButton(QMessageBox.Yes)
    trace_string = "".join(traceback.format_exception(type_, value, tback))
    msg_box.setDetailedText(trace_string)
    msg_box.setIcon(QMessageBox.Critical)
    if msg_box.exec() == QMessageBox.Yes:
        sentry_sdk.capture_message("Uncaught exception" +  trace_string)
    # then call the default handler
    sys.__excepthook__(type, value, tback)
