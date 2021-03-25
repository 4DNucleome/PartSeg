from qtpy.QtCore import QThread
from qtpy.QtWidgets import QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton

from PartSeg.common_backend.progress_thread import ExecuteFunctionThread

from ..common_backend.progress_thread import ProgressTread


class WaitingDialog(QDialog):
    def __init__(self, thread: QThread, text="", parent=None, exception_hook=None):
        super().__init__(parent)
        label = QLabel(text)
        self.exception_hook = exception_hook
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setDisabled(True)
        layout = QHBoxLayout()
        layout.addWidget(label)
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel_btn)
        self.cancel_btn.clicked.connect(self.close)
        thread.finished.connect(self.accept)
        self.thread_to_wait = thread
        self.setLayout(layout)
        if isinstance(thread, ProgressTread):
            thread.range_changed.connect(self.progress.setRange)
            thread.step_changed.connect(self.progress.setValue)
            thread.error_signal.connect(self.error_catch)

    def error_catch(self, error):
        # print(self.thread() == QApplication.instance().thread(), error, isinstance(error, Exception))
        self.close()
        if self.exception_hook:
            self.exception_hook(error)
        else:
            raise error

    def exec(self):
        self.thread_to_wait.start()
        return super().exec_()


class ExecuteFunctionDialog(WaitingDialog):
    def __init__(self, fun, args=None, kwargs=None, text="", parent=None, exception_hook=None):
        thread = ExecuteFunctionThread(fun, args, kwargs)
        super().__init__(thread, text=text, parent=parent, exception_hook=exception_hook)

    def get_result(self):
        return self.thread_to_wait.result
