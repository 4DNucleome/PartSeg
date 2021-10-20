import qtawesome as qta
from qtpy.QtGui import QIcon, QPainter, QPaintEvent
from qtpy.QtWidgets import QCheckBox


class LockCheckBox(QCheckBox):
    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        lock: QIcon = qta.icon("fa5s.lock") if self.isChecked() else qta.icon("fa5s.lock-open")
        lock.paint(painter, rect)
