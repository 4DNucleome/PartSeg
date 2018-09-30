from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import QDialog, QProgressBar, QPushButton, QHBoxLayout


class WaitingDialog(QDialog):
    def __init__(self, thread: QThread, parent=None):
        super().__init__(parent)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setDisabled(True)
        layout = QHBoxLayout()
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel_btn)
        self.cancel_btn.clicked.connect(self.close)
        thread.finished.connect(self.accept)
        self.thread_to_wait = thread
        self.setLayout(layout)

    def exec(self):
        self.thread_to_wait.start()
        return super().exec()

