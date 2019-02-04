from qtpy.QtWidgets import QCheckBox
from qtpy.QtGui import QPaintEvent, QPainter
import qtawesome as qta

lock_close = u"\U0001F512"
lock_open = u"\U0001F513"

class LockCheckBox(QCheckBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.lock = qta.icon("fa5s.lock")
        self.open = qta.icon("fa5s.lock-open")

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        if self.isChecked():
            #lock = self.lock
            lock = lock_close
        else:
            #lock = self.open
            lock = lock_open
        #lock.paint(painter, rect)
        painter.drawText(rect.bottomLeft(), lock)
