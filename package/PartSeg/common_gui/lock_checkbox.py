import qtawesome as qta
from qtpy.QtGui import QPainter, QPaintEvent
from qtpy.QtWidgets import QCheckBox


class LockCheckBox(QCheckBox):
    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        color = painter.pen().color()
        lock = qta.icon("fa5s.lock", color=color) if self.isChecked() else qta.icon("fa5s.lock-open", color=color)
        lock.paint(painter, rect)
