"""
This module contains widgets to create and manage labels scheme
"""
from copy import deepcopy

import numpy as np
from qtpy.QtCore import Qt, Signal, Slot
from qtpy.QtGui import QColor, QMouseEvent, QPainter, QPaintEvent
from qtpy.QtWidgets import (
    QApplication,
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from PartSegCore.custom_name_generate import custom_name_generate

from ..common_backend.base_settings import ViewSettings
from .icon_selector import IconSelector
from .numpy_qimage import NumpyQImage

_icon_selector = IconSelector()


def add_alpha_channel(colors):
    if colors.shape[1] == 4:
        return colors
    new_label = np.zeros((colors.shape[0], 4), dtype=np.uint8)
    new_label[:, :3] = colors
    new_label[:, 3] = 255
    return new_label


class _LabelShow(QWidget):
    def __init__(self, label: np.ndarray):
        super().__init__()
        self.image = None
        self.set_labels(label)

    def set_labels(self, label):
        if label.ndim != 2 and label.shape[1] not in (3, 4):
            raise ValueError("Wrong array shape")
        label = add_alpha_channel(label)
        self.image = NumpyQImage(label.reshape((1,) + label.shape))
        self.repaint()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        rect = self.rect()
        painter.drawRect(rect)
        painter.drawImage(rect, self.image)


class LabelShow(QWidget):
    """Present single label color scheme"""

    remove_labels = Signal(str)
    edit_labels = Signal([str, list], [list])
    selected = Signal(str)

    def __init__(self, name: str, label: list, removable, parent=None):
        super().__init__(parent)
        self.label = label
        self.name = name
        self.removable = removable

        self.radio_btn = QRadioButton()

        self.label_show = _LabelShow(np.array(label, dtype=np.uint8))

        self.remove_btn = QToolButton()
        self.remove_btn.setIcon(_icon_selector.close_icon)

        self.edit_btn = QToolButton()
        self.edit_btn.setIcon(_icon_selector.edit_icon)

        if removable:
            self.remove_btn.setToolTip("Remove colormap")
        else:
            self.remove_btn.setToolTip("This colormap is protected")

        self.edit_btn.setToolTip("Create new label schema base on this")

        layout = QHBoxLayout()
        layout.addWidget(self.radio_btn)
        layout.addWidget(self.label_show, 1)
        layout.addWidget(self.remove_btn)
        layout.addWidget(self.edit_btn)
        self.remove_btn.setEnabled(removable)
        self.setLayout(layout)
        self.remove_btn.clicked.connect(self.remove_fun)
        self.edit_btn.clicked.connect(self.edit_fun)
        self.radio_btn.clicked.connect(self.selected_fun)

    def set_checked(self, val):
        self.radio_btn.setChecked(val)
        if self.removable:
            self.remove_btn.setDisabled(val)

    @Slot()
    def remove_fun(self):
        if self.remove_btn.isEnabled():
            self.remove_labels.emit(self.name)

    @Slot()
    def edit_fun(self):
        self.edit_labels.emit(self.name, deepcopy(self.label))
        self.edit_labels[list].emit(deepcopy(self.label))

    @Slot(bool)
    def selected_fun(self):
        self.selected.emit(self.name)


class LabelChoose(QWidget):
    edit_signal = Signal([str, list], [list])

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        layout = QVBoxLayout()
        self.setLayout(layout)

    @Slot(str)
    def change_scheme(self, name):
        self.settings.current_labels = name
        for i in range(self.layout().count()):
            el = self.layout().itemAt(i)
            if el.widget():
                if el.widget().name != name:
                    el.widget().set_checked(False)
                else:
                    el.widget().set_checked(True)

    @Slot(str)
    def remove(self, name: str):
        del self.settings.label_color_dict[name]
        self.refresh()

    def refresh(self):
        for _ in range(self.layout().count()):
            el = self.layout().takeAt(0)
            if el.widget():
                w: LabelShow = el.widget()
                w.selected.disconnect()
                w.remove_labels.disconnect()
                w.edit_labels[str, list].disconnect()
                w.edit_labels[list].disconnect()
                el.widget().deleteLater()

        chosen_name = self.settings.current_labels
        for name, (val, removable) in self.settings.label_color_dict.items():
            label = LabelShow(name, val, removable, self)
            if name == chosen_name:
                label.set_checked(True)
            label.selected.connect(self.change_scheme)
            label.remove_labels.connect(self.remove)
            label.edit_labels[list].connect(self.edit_signal[list].emit)
            label.edit_labels[str, list].connect(self.edit_signal[str, list].emit)
            self.layout().addWidget(label)
        self.layout().addStretch(1)

    def showEvent(self, _):
        self.refresh()


class ColorShow(QLabel):
    """Widget to show chosen color. Change mouse cursor when above."""

    def __init__(self, color, parent=None):
        super().__init__(parent)
        self.color = color
        self._qcolor = QColor(*color)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(event.rect(), self._qcolor)

    def enterEvent(self, QEvent):  # pylint: disable=R0201
        QApplication.setOverrideCursor(Qt.DragMoveCursor)

    def leaveEvent(self, QEvent):  # pylint: disable=R0201
        QApplication.restoreOverrideCursor()

    def set_color(self, color):
        self.color = color
        self._qcolor = QColor(*color)
        self.repaint()


class LabelEditor(QWidget):
    """Widget for create label scheme."""

    def __init__(self, settings: ViewSettings):
        super().__init__()
        self.settings = settings
        self.color_list = []
        self.chosen = None
        self.prohibited_names = set(self.settings.label_color_dict.keys())  # Prohibited name is added to reduce
        # probability of colormap cache collision

        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.Widget)
        self.color_picker.setOptions(QColorDialog.DontUseNativeDialog | QColorDialog.NoButtons)
        self.add_color_btn = QPushButton("Add color")
        self.add_color_btn.clicked.connect(self.add_color)
        self.remove_color_btn = QPushButton("Remove last color")
        self.remove_color_btn.clicked.connect(self.remove_color)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save)

        self.color_layout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(self.color_picker)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_color_btn)
        btn_layout.addWidget(self.remove_color_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        layout.addLayout(self.color_layout)
        self.setLayout(layout)

    @Slot(list)
    def set_colors(self, colors: list):
        for _ in range(self.color_layout.count()):
            el = self.color_layout.takeAt(0)
            if el.widget():
                el.widget().deleteLater()
        for color in colors:
            self.color_layout.addWidget(ColorShow(color, self))

    def remove_color(self):
        if self.color_layout.count():
            el = self.color_layout.takeAt(self.color_layout.count() - 1)
            el.widget().deleteLater()

    def add_color(self):
        color = self.color_picker.currentColor()
        self.color_layout.addWidget(ColorShow([color.red(), color.green(), color.blue()], self))

    def get_colors(self):
        count = self.color_layout.count()
        return [self.color_layout.itemAt(i).widget().color for i in range(count)]

    def save(self):
        count = self.color_layout.count()
        if not count:
            return
        rand_name = custom_name_generate(self.prohibited_names, self.settings.label_color_dict)
        self.prohibited_names.add(rand_name)
        self.settings.label_color_dict[rand_name] = self.get_colors()

    def mousePressEvent(self, e: QMouseEvent):
        child = self.childAt(e.pos())
        if not isinstance(child, ColorShow):
            self.chosen = None
            return
        self.chosen = child

    def mouseMoveEvent(self, e: QMouseEvent):
        if self.chosen is None:
            return
        index = self.color_layout.indexOf(self.chosen)
        index2 = int(e.x() / self.width() * self.color_layout.count() + 0.5)
        if index2 != index:
            self.color_layout.insertWidget(index2, self.chosen)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self.chosen = None
