from enum import Enum
from typing import List

from qtpy.QtCore import QRect, QSize
from qtpy.QtWidgets import QLayout, QLayoutItem


class LayoutPosition(Enum):
    left = 0
    right = 1


class EqualColumnLayout(QLayout):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._item_list: List[QLayoutItem] = []

    def addItem(self, item: QLayoutItem):
        self._item_list.append(item)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self.calc_position(rect)

    def sizeHint(self) -> QSize:
        height, width = 0, 0
        for el in self._item_list:
            if el.widget() and el.widget().isHidden():
                continue
            ob_size: QSize = el.sizeHint()
            height = max(height, ob_size.height())
            width += ob_size.width()
        return QSize(width, height)

    def itemAt(self, p_int):
        try:
            return self._item_list[p_int]
        except IndexError:
            return None

    def takeAt(self, p_int):
        try:
            return self._item_list.pop(p_int)
        except IndexError:
            return None

    def count(self) -> int:
        return len(self._item_list)

    def calc_position(self, rect: QRect):
        columns = sum(1 for el in self._item_list if el.widget() and el.widget().isVisible())

        if columns == 0:
            return
        element_width = rect.width() // columns
        x = rect.x()
        for el in self._item_list:
            if el.widget() and el.widget().isHidden():
                continue
            el.setGeometry(QRect(x, rect.y(), element_width, rect.height()))
            x += element_width
