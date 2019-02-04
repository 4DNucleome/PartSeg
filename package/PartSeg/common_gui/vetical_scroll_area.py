from qtpy import QtGui
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QScrollArea


class VerticalScrollArea(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        # self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

    def resizeEvent(self, a0: QtGui.QResizeEvent):
        if self.widget() and self.width() > 0:
            self.widget().setMinimumWidth(self.width() - self.verticalScrollBar().width())
        super().resizeEvent(a0)
