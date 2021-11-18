import typing
from contextlib import suppress

import qtawesome as qta
from qtpy import QtGui
from qtpy.QtCore import QRect, Qt
from qtpy.QtGui import QFont, QFontMetrics, QPainter
from qtpy.QtWidgets import QCheckBox, QWidget


class CollapseCheckbox(QCheckBox):
    """
    Check box for hide widgets. It is painted as:
    ▶, {info_text}, line

    If triangle is ▶ then widgets are hidden
    If triangle is ▼ then widgets are shown

    :param info_text: optional text to be show
    """

    def __init__(self, info_text: str = "", parent: typing.Optional[QWidget] = None):
        super().__init__(info_text or "-", parent)
        self.hide_list = []
        self.stateChanged.connect(self.hide_element)

        metrics = QFontMetrics(QFont())
        self.text_size = metrics.size(Qt.TextSingleLine, info_text)
        self.info_text = info_text

    def add_hide_element(self, val: QWidget):
        """
        Add widget which visibility should be controlled by CollapseCheckbox
        """
        self.hide_list.append(val)

    def remove_hide_element(self, val: QWidget):
        """
        Stop controlling widget visibility by CollapseCheckbox
        """
        with suppress(ValueError):
            self.hide_list.remove(val)

    def hide_element(self, a0: int):
        for el in self.hide_list:
            el.setHidden(bool(a0))

    def paintEvent(self, event: QtGui.QPaintEvent):
        painter = QPainter(self)
        color = painter.pen().color()
        painter.save()
        rect = self.rect()
        top = int(rect.height() - (self.text_size.height() / 2))
        painter.drawText(rect.height() + 5, top, self.info_text)
        if self.isChecked():
            icon = qta.icon("fa5s.caret-right", color=color)
        else:
            icon = qta.icon("fa5s.caret-down", color=color)
        icon.paint(painter, QRect(0, -self.height() / 4, self.height(), self.height()))
        painter.restore()
