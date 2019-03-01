import sys

from qtpy.QtWidgets import QDialog, QPushButton, QTextEdit, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit
import traceback
from ..utils import state_store
import sentry_sdk

from PartSeg import __version__


class ErrorDialog(QDialog):
    def __init__(self, exception: Exception, description: str, additional_notes: str = "", traceback_summary=None):
        super().__init__()
        self.exception = exception
        self.additional_notes = additional_notes
        self.send_report_btn = QPushButton("Send information")
        self.send_report_btn.setDisabled(not state_store.report_errors)
        self.cancel_btn = QPushButton("Cancel")
        self.error_description = QTextEdit()
        if traceback_summary is None:
            self.error_description.setText("".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)))
        elif isinstance(traceback_summary, traceback.StackSummary):
            self.error_description.setText("".join(traceback_summary.format()))
        self.error_description.append(str(exception))
        self.error_description.setReadOnly(True)
        self.additional_info = QTextEdit()
        self.contact_info = QLineEdit()

        self.cancel_btn.clicked.connect(self.reject)
        self.send_report_btn.clicked.connect(self.send_information)

        layout = QVBoxLayout()
        self.desc = QLabel(description)
        self.desc.setWordWrap(True)
        layout.addWidget(self.desc)
        layout.addWidget(self.error_description)
        layout.addWidget(QLabel("Contact information"))
        layout.addWidget(self.contact_info)
        layout.addWidget(QLabel("Additional information from user:"))
        layout.addWidget(self.additional_info)
        if not state_store.report_errors:
            layout.addWidget(QLabel("Sending reports was disabled by runtime flag. "
                                    "You can report it manually by creating report on"
                                    "https://github.com/4DNucleome/PartSeg/issues"))
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.send_report_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def exec(self):
        if not state_store.show_error_dialog:
            sys.__excepthook__(type(self.exception), self.exception, self.exception.__traceback__)
            return False
        super().exec()

    def send_information(self):
        text = self.desc.text() + "\n\nVersion: " + __version__ + "\n"
        if len(self.additional_notes) > 0:
            text += "Additional notes: " + self.additional_notes + "\n"
        text += self.error_description.toPlainText() + "\n\n"
        if len(self.additional_info.toPlainText()) > 0:
            text += "\nUser information:\n" + self.additional_info.toPlainText()
        if len(self.contact_info.text()) > 0:
            text += "\nContact: " + self.contact_info.text()
        sentry_sdk.capture_message(text, level='error')
        self.accept()
