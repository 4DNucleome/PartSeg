import typing
from math import sqrt

from qtpy import QtGui
from qtpy.QtCore import QLineF, QPointF, Qt
from qtpy.QtGui import QFont, QFontMetrics, QPainter, QPolygonF
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
        try:
            self.hide_list.remove(val)
        except ValueError:
            pass

    def hide_element(self, a0: int):
        for el in self.hide_list:
            el.setHidden(bool(a0))

    def paintEvent(self, event: QtGui.QPaintEvent):
        border_distance = 5
        rect = self.rect()
        mid = rect.y() + rect.height() / 2
        line_begin = QPointF(rect.height() + 10 + self.text_size.width(), mid)
        line_end = QPointF(rect.width() + rect.x() - 5, mid)
        triangle = QPolygonF()
        side_length = rect.height() - 2 * border_distance
        triangle_height = side_length * sqrt(3) / 2
        start_point = QPointF(rect.x() + border_distance, rect.y() + border_distance)

        if self.isChecked():
            triangle.append(start_point)
            triangle.append(start_point + QPointF(0, side_length))
            triangle.append(start_point + QPointF(triangle_height, side_length / 2))
        else:
            triangle.append(start_point)
            triangle.append(start_point + QPointF(side_length, 0))
            triangle.append(start_point + QPointF(side_length / 2, triangle_height))
        painter = QPainter(self)

        painter.setBrush(Qt.black)
        top = int(rect.height() - (self.text_size.height() / 2))
        painter.drawText(rect.height() + 5, top, self.info_text)
        painter.drawPolygon(triangle, Qt.WindingFill)
        painter.drawLine(QLineF(line_begin, line_end))
