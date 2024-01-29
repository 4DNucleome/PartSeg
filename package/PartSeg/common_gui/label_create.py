"""
This module contains widgets to create and manage labels scheme
"""

import json
import typing
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
from fonticon_fa6 import FA6S
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
from superqt.fonticon import setTextIcon

from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.custom_load_dialog import PLoadDialog
from PartSeg.common_gui.custom_save_dialog import PSaveDialog
from PartSeg.common_gui.icon_selector import IconSelector
from PartSeg.common_gui.numpy_qimage import NumpyQImage
from PartSegCore.custom_name_generate import custom_name_generate
from PartSegCore.io_utils import IO_LABELS_COLORMAP, LoadBase, SaveBase
from PartSegCore.utils import BaseModel

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
            raise ValueError("Wrong array shape")  # pragma: no cover
        label = add_alpha_channel(label)
        self.image = NumpyQImage(label.reshape((1, *label.shape)))
        self.repaint()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        rect = self.rect()
        painter.drawRect(rect)
        painter.drawImage(rect, self.image)


class LabelShow(QWidget):
    """Present single label color scheme"""

    remove_labels = Signal(str)
    edit_labels = Signal(list)
    edit_labels_with_name = Signal(str, list)
    selected = Signal(str)

    def __init__(self, name: str, label: List[Sequence[float]], removable, parent=None):
        super().__init__(parent)
        self.label = label
        self.name = name
        self.removable = removable

        self.radio_btn = QRadioButton()

        self.label_show = _LabelShow(np.array(label, dtype=np.uint8))

        self.remove_btn = QToolButton()
        setTextIcon(self.remove_btn, FA6S.trash_can, 16)

        self.edit_btn = QToolButton()
        setTextIcon(self.edit_btn, FA6S.pen, 16)

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
        self.radio_btn.toggled.connect(self.selected_fun)

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
        self.edit_labels_with_name.emit(self.name, deepcopy(self.label))
        self.edit_labels.emit(deepcopy(self.label))

    @Slot(bool)
    def selected_fun(self):
        if self.radio_btn.isChecked():
            self.selected.emit(self.name)


class LabelChoose(QWidget):
    edit_signal = Signal(list)
    edit_with_name_signal = Signal(str, list)

    def __init__(self, settings, parent=None):
        super().__init__(parent)
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
        if name in self.settings.label_color_dict:
            del self.settings.label_color_dict[name]
        self.refresh()

    def refresh(self):
        for _ in range(self.layout().count()):
            el = self.layout().takeAt(0)
            if el.widget():
                w: LabelShow = el.widget()
                w.selected.disconnect()
                w.remove_labels.disconnect()
                w.edit_labels_with_name.disconnect()
                w.edit_labels.disconnect()
                el.widget().deleteLater()

        chosen_name = self.settings.current_labels
        for name, (val, removable) in self.settings.label_color_dict.items():
            label = self._label_show(name, val, removable)
            if name == chosen_name:
                label.set_checked(True)
            label.selected.connect(self.change_scheme)
            label.remove_labels.connect(self.remove)
            label.edit_labels.connect(self.edit_signal.emit)
            label.edit_labels_with_name.connect(self.edit_with_name_signal.emit)
            self.layout().addWidget(label)
        self.layout().addStretch(1)

    def _label_show(self, name: str, label: List[Sequence[float]], removable) -> LabelShow:
        return LabelShow(name, label, removable, self)

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

    def enterEvent(self, _event):  # pylint: disable=no-self-use
        QApplication.setOverrideCursor(Qt.CursorShape.DragMoveCursor)

    def leaveEvent(self, _event):  # pylint: disable=no-self-use
        QApplication.restoreOverrideCursor()

    def set_color(self, color):
        self.color = color
        self._qcolor = QColor(*color)
        self.repaint()


class LabelEditor(QWidget):
    """Widget for create label scheme."""

    def __init__(self, settings: ViewSettings, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.color_list = []
        self.chosen = None
        self.prohibited_names = set(self.settings.label_color_dict.keys())  # Prohibited name is added to reduce
        # probability of colormap cache collision

        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.WindowType.Widget)
        self.color_picker.setOptions(
            QColorDialog.ColorDialogOption.DontUseNativeDialog | QColorDialog.ColorDialogOption.NoButtons
        )
        self.add_color_btn = QPushButton("Add color")
        self.add_color_btn.clicked.connect(self.add_color)
        self.remove_color_btn = QPushButton("Remove last color")
        self.remove_color_btn.clicked.connect(self.remove_color)
        self.save_btn = QPushButton("Save")
        self.save_btn.clicked.connect(self.save)
        self.import_btn = QPushButton("Import")
        self.import_btn.clicked.connect(self._import_action)
        self.export_btn = QPushButton("Export")
        self.export_btn.clicked.connect(self._export_action)

        self.color_layout = QHBoxLayout()
        layout = QVBoxLayout()
        layout.addWidget(self.color_picker)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.add_color_btn)
        btn_layout.addWidget(self.remove_color_btn)
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)
        layout.addLayout(self.color_layout)
        self.setLayout(layout)

    def _import_action(self):
        dial = PLoadDialog(LabelsLoad, settings=self.settings, path=IO_LABELS_COLORMAP)
        if dial.exec_():
            res = dial.get_result()
            self.set_colors("", res.load_class.load(res.load_location))

    def _export_action(self):
        if not self.color_layout.count():
            return
        self.get_colors()
        dial = PSaveDialog(
            LabelsSave,
            settings=self.settings,
            path=IO_LABELS_COLORMAP,
        )
        if dial.exec_():
            res = dial.get_result()
            res.save_class.save(res.save_destination, self.get_colors(), res.parameters)

    @Slot(str, list)
    def set_colors(self, name: str, colors: list):
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

    def get_colors(self) -> List[List[int]]:
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


class LabelsLoad(LoadBase):
    __argument_class__ = BaseModel

    @classmethod
    def get_name(cls) -> str:
        return "Labels json (*.label.json)"

    @classmethod
    def load(
        cls,
        load_locations: List[Union[str, BytesIO, Path]],
        range_changed: Optional[Callable[[int, int], Any]] = None,
        step_changed: Optional[Callable[[int], Any]] = None,
        metadata: Optional[dict] = None,
    ) -> List[List[float]]:
        with open(load_locations[0]) as f_p:
            return json.load(f_p)

    @classmethod
    def get_short_name(cls):
        return "label_json"


class LabelsSave(SaveBase):
    __argument_class__ = BaseModel

    @classmethod
    def get_short_name(cls):
        return "label_json"

    @classmethod
    def save(
        cls,
        save_location: typing.Union[str, BytesIO, Path],
        project_info,
        parameters: Optional[dict] = None,
        range_changed=None,
        step_changed=None,
    ):
        with open(save_location, "w") as f_p:
            json.dump(project_info, f_p)

    @classmethod
    def get_name(cls) -> str:
        return "Labels json (*.label.json)"
