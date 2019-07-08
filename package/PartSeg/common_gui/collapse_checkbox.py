from math import sqrt

from qtpy import QtGui
from qtpy.QtCore import QPointF, Qt, QLineF
from qtpy.QtGui import QPolygonF, QPainter, QFontMetrics, QFont
from qtpy.QtWidgets import QCheckBox, QWidget


class CollapseCheckbox(QCheckBox):
    """
    :type hide_list: typing.List[QWidget]
    """
    def __init__(self, info_text="", parent: QWidget = None):
        super().__init__(info_text if info_text else "-", parent)
        self.hide_list = []
        self.stateChanged.connect(self.hide_element)

        metrics = QFontMetrics(QFont())
        self.text_size = metrics.size(Qt.TextSingleLine, info_text)
        self.info_text = info_text

    def add_hide_element(self, val):
        self.hide_list.append(val)

    def remove_hide_element(self, val):
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
        mid = rect.y() + rect.height()/2
        line_begin = QPointF(rect.height() + 10 + self.text_size.width(), mid)
        line_end = QPointF(rect.width() + rect.x() - 5, mid)
        triangle = QPolygonF()
        side_length = rect.height() - 2 * border_distance
        triangle_height = side_length * sqrt(3) / 2
        start_point = QPointF(rect.x() + border_distance, rect.y() + border_distance)

        if self.isChecked():
            triangle.append(start_point)
            triangle.append(start_point + QPointF(0, side_length))
            triangle.append(start_point + QPointF(triangle_height, side_length/2))
        else:
            triangle.append(start_point)
            triangle.append(start_point + QPointF(side_length, 0))
            triangle.append(start_point + QPointF(side_length / 2, triangle_height))
        painter = QPainter(self)

        painter.setBrush(Qt.black)
        top = rect.height() - (self.text_size.height() / 2)
        painter.drawText(rect.height() + 5, top, self.info_text)
        painter.drawPolygon(triangle, Qt.WindingFill)
        painter.drawLine(QLineF(line_begin, line_end))
