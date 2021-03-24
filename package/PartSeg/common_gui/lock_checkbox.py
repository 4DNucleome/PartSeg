from qtpy.QtCore import QPoint
from qtpy.QtGui import QFont, QPainter, QPaintEvent
from qtpy.QtWidgets import QCheckBox

lock_close = "\U0001F512"
lock_open = "\U0001F513"


class LockCheckBox(QCheckBox):
    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        painter.save()
        lock = lock_close if self.isChecked() else lock_open
        painter.setFont(QFont("Symbola"))
        painter.drawText(rect.bottomLeft() + QPoint(0, -2), lock)
        painter.restore()
