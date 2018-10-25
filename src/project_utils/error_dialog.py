from PyQt5.QtWidgets import QDialog, QPushButton, QTextEdit, QHBoxLayout, QVBoxLayout, QLabel
import traceback
import sentry_sdk

sentry_sdk.init("https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")

class ErrorDialog(QDialog):
    def __init__(self, exception: Exception, description: str, additional_notes: str = ""):
        super().__init__()
        self.additional_notes = additional_notes
        self.send_report_btn = QPushButton("Send information")
        self.cancel_btn = QPushButton("Cancel")
        self.error_description = QTextEdit()
        self.error_description.setText("".join(traceback.format_tb(exception.__traceback__)))
        self.error_description.setReadOnly(True)
        self.additional_info = QTextEdit()

        self.cancel_btn.clicked.connect(self.reject)
        self.send_report_btn.clicked.connect(self.send_information)

        layout = QVBoxLayout()
        self.desc = QLabel(description)
        self.desc.setWordWrap(True)
        layout.addWidget(self.desc)
        layout.addWidget(self.error_description)
        layout.addWidget(QLabel("Additional information from user:"))
        layout.addWidget(self.additional_info)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.cancel_btn)
        btn_layout.addWidget(self.send_report_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def send_information(self):
        text = self.desc.text() + "\n"
        if len(self.additional_notes) > 0:
            text += "Additional notes: " + self.additional_notes +"\n"
        text +=  self.error_description.toPlainText() + "\n\n"
        if len(self.additional_info.toPlainText()) > 0:
            text += "User information:\n" + self.additional_info.toPlainText()
        sentry_sdk.capture_message(text)
        self.accept()

