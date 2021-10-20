import qtawesome as qta
from qtpy.QtGui import QIcon, QPainter, QPaintEvent
from qtpy.QtWidgets import QCheckBox

lock_close = "\U0001F512"
lock_open = "\U0001F513"


class LockCheckBox(QCheckBox):
    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        lock: QIcon = qta.icon("fa5s.lock") if self.isChecked() else qta.icon("fa5s.lock-open")
        lock_pixmap = lock.pixmap(rect.size())
        painter.drawPixmap(0, 0, lock_pixmap)
