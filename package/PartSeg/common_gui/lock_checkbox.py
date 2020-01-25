from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QCheckBox
from qtpy.QtGui import QPaintEvent, QPainter, QFont

lock_close = "\U0001F512"
lock_open = "\U0001F513"


class LockCheckBox(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        painter.save()
        if self.isChecked():
            lock = lock_close
        else:
            lock = lock_open
        painter.setFont(QFont("Symbola"))
        painter.drawText(rect.bottomLeft() + QPoint(0, -2), lock)
        painter.restore()
