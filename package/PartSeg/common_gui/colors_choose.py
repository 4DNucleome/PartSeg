import typing
from itertools import count

import numpy as np

from qtpy.QtCore import Signal
from qtpy.QtGui import QImage
from qtpy.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QPushButton, QHBoxLayout

from .channel_control import ColorPreview
from .flow_layout import FlowLayout
from .vetical_scroll_area import VerticalScrollArea
from ..utils.color_image import color_image
from ..project_utils_qt.settings import ViewSettings


class CheckBoxWithMouseSignal(QCheckBox):
    mouse_over = Signal(str)

    def mouseMoveEvent(self, _):
        self.mouse_over.emit(self.text())


class ColorSelector(QWidget):
    def __init__(self, settings: ViewSettings, control_names: typing.List[str], parent=None):
        super().__init__(parent)
        self.image = None
        self.preview = ColorPreview(self)
        self.preview.setMinimumHeight(40)
        self.control_names = control_names
        self.settings = settings
        self.color_widget_list: typing.List[QCheckBox] = []
        self.accept_button = QPushButton("Save")
        self.reset_button = QPushButton("Reset")
        self.accept_button.clicked.connect(self.save)
        self.reset_button.clicked.connect(self.reset)
        self.mark_all_btn = QPushButton("Mark all")
        self.un_mark_all_btn = QPushButton("Clear selection")
        self.mark_all_btn.clicked.connect(self.mark_all)
        self.un_mark_all_btn.clicked.connect(self.un_mark_all)
        self.current_color = ""
        self.scroll = VerticalScrollArea()
        widget = QWidget()

        layout = QVBoxLayout()

        btn_layout1 = QHBoxLayout()
        btn_layout1.addWidget(self.mark_all_btn)
        btn_layout1.addWidget(self.un_mark_all_btn)

        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.accept_button)
        btn_layout2.addWidget(self.reset_button)

        layout.addWidget(self.preview)
        layout.addLayout(btn_layout1)

        self.flow_layout = FlowLayout()
        self._set_colormaps()
        widget.setLayout(self.flow_layout)
        self.scroll.setWidget(widget)
        layout.addWidget(self.scroll)
        layout.addLayout(btn_layout2)
        self.setLayout(layout)

    def mark_all(self):
        for el in self.color_widget_list:
            el.setChecked(True)

    def un_mark_all(self):
        for el in self.color_widget_list:
            if el.isEnabled():
                el.setChecked(False)

    def save(self):
        res = []
        for el in self.color_widget_list:
            if el.isChecked():
                res.append(el.text())
        self.settings.chosen_colormap = res

    def reset(self):
        blocked = []
        for el in self.control_names:
            data = self.settings.get_from_profile(el)
            for i in count(0):
                if f"cmap{i}" in data:
                    blocked.append(data[f"cmap{i}"])
                else:
                    break
        blocked = set(blocked)
        chosen_colormap = set(self.settings.chosen_colormap)

        for el in self.color_widget_list:
            el.setChecked(el.text() in chosen_colormap)
            el.setDisabled(el.text() in blocked)

    def _set_colormaps(self):
        colormap_list = self.settings.available_colormaps
        for el in colormap_list:
            check = CheckBoxWithMouseSignal(el)
            check.mouse_over.connect(self.mouse_on_map)
            self.color_widget_list.append(check)
            self.flow_layout.addWidget(check)

    def mouse_on_map(self, val):
        if val != self.current_color:
            self.current_color = val
            img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [val], [(0, 256)])
            self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
            self.preview.repaint()

    def showEvent(self, _):
        self.reset()
