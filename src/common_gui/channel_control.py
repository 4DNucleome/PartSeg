from PyQt5.QtWidgets import QWidget, QSpinBox, QCheckBox, QGridLayout, QLabel, QHBoxLayout, QComboBox
from PyQt5.QtGui import QImage, QShowEvent, QPaintEvent, QPainter, QPen, QMouseEvent, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from common_gui.collapse_checkbox import CollapseCheckbox
from project_utils.color_image import color_image
import typing


class ColorPreview(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)

        painter.drawImage(rect, self.parent().image)


class ChannelWidget(QWidget):
    clicked = pyqtSignal(int)

    def __init__(self, id: int, color: str):
        super().__init__()
        self.id = id
        self.active = False
        self.color = color
        self.chosen = QCheckBox(self)
        self.chosen.setChecked(True)
        layout = QHBoxLayout()
        layout.addWidget(self.chosen, 0)
        layout.addWidget(QLabel(""), 1)
        img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [self.color], [(0, 256)])
        self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
        self.setLayout(layout)

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)

        painter.drawImage(rect, self.image)
        if self.active:
            pen = QPen()
            pen.setWidth(5)
            pen1 = painter.pen()
            painter.setPen(pen)
            painter.drawRect(event.rect())
            painter.setPen(pen1)
        super().paintEvent(event)

    def set_active(self, val=True):
        self.active = val
        self.repaint()

    def set_inactive(self, val=True):
        self.active = not val
        self.repaint()

    def mousePressEvent(self, a0: QMouseEvent):
        self.clicked.emit(self.id)
        print(self.color)


class MyComboBox(QComboBox):
    hide_popup = pyqtSignal()

    def hidePopup(self):
        super().hidePopup()
        self.hide_popup.emit()


class ChannelControl(QWidget):
    """
    :Type channels_widgets: typing.List[ChannelWidget]
    """

    coloring_update = pyqtSignal(list, bool)

    def __init__(self, settings, parent=None, flags=Qt.WindowFlags()):
        super().__init__(parent, flags)
        self._settings = settings
        self.current_channel = 0
        self.colormap_chose = MyComboBox()
        self.colormap_chose.addItems(self._settings.chosen_colormap)
        self.colormap_chose.highlighted[str].connect(self.change_color_preview)
        self.colormap_chose.hide_popup.connect(self.change_closed)
        self.colormap_chose.activated[str].connect(self.change_color)
        self.channel_preview_widget = ColorPreview(self)
        # self.channel_preview_widget.setPixmap(QPixmap.fromImage(self.channel_preview))
        self.minimum_value = QSpinBox(self)
        self.maximum_value = QSpinBox(self)
        self.fixed = QCheckBox("Fix range")
        self.collapse_widget = CollapseCheckbox()
        self.collapse_widget.add_hide_element(self.minimum_value)
        self.collapse_widget.add_hide_element(self.maximum_value)
        self.collapse_widget.add_hide_element(self.fixed)
        self.channels_widgets = []
        self.channels_layout = QHBoxLayout()
        layout = QGridLayout()
        layout.addLayout(self.channels_layout, 0, 0, 1, 4)
        layout.addWidget(self.channel_preview_widget, 2, 0, 1, 2)
        layout.addWidget(self.colormap_chose, 2, 2, 1, 2)
        layout.addWidget(self.collapse_widget, 3, 0, 1, 4)
        label1 = QLabel("Min bright")
        layout.addWidget(label1, 4, 0)
        layout.addWidget(self.minimum_value, 4, 1)
        label2 = QLabel("Max bright")
        layout.addWidget(label2, 4, 2)
        layout.addWidget(self.maximum_value, 4, 3)
        layout.addWidget(self.fixed, 5, 0, 1, 2)
        self.collapse_widget.add_hide_element(label1)
        self.collapse_widget.add_hide_element(label2)
        self.setLayout(layout)
        self._settings.image_changed.connect(self.update_channels_list)

    def change_chanel(self, id):
        self.channels_widgets[self.current_channel].set_inactive()
        self.channels_widgets[id].set_active()
        self.current_channel = id
        self.image = self.channels_widgets[id].image
        self.colormap_chose.setCurrentText(self.channels_widgets[id].color)
        self.channel_preview_widget.repaint()

    def change_closed(self):
        text = self.colormap_chose.currentText()
        self.change_color_preview(text)

    def change_color_preview(self, value):
        img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [value], [(0, 256)])
        self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
        self.channel_preview_widget.repaint()

    def change_color(self, value):
        widget = self.channels_widgets[self.current_channel]
        widget.deleteLater()
        new_widget = ChannelWidget(self.current_channel, value)
        new_widget.clicked.connect(self.change_chanel)
        new_widget.chosen.stateChanged.connect(self.send_info_wrap)
        self.channels_layout.replaceWidget(widget, new_widget)
        self.channels_widgets[self.current_channel] = new_widget
        new_widget.set_active()
        widget.clicked.disconnect()
        widget.chosen.stateChanged.disconnect()
        self.change_color_preview(value)
        self._settings.colors[self.current_channel] = str(value)
        self.send_info()

    def update_channels_list(self):
        channels_num = self._settings.channels
        for el in self.channels_widgets:
            self.channels_layout.removeWidget(el)
            el.clicked.disconnect()
            el.chosen.stateChanged.disconnect()
            el.deleteLater()
        self.channels_widgets = []
        for i in range(channels_num):
            self.channels_widgets.append(ChannelWidget(i, self._settings.colors[i]))
            self.channels_layout.addWidget(self.channels_widgets[-1])
            self.channels_widgets[-1].clicked.connect(self.change_chanel)
            self.channels_widgets[-1].chosen.stateChanged.connect(self.send_info_wrap)
        self.channels_widgets[0].set_active()
        self.current_channel = 0
        self.image = self.channels_widgets[0].image
        self.colormap_chose.setCurrentText(self.channels_widgets[0].color)
        self.send_info(True)

    def send_info_wrap(self):
        self.send_info()

    def send_info(self, new_image=False):
        channels_num = len(self.channels_widgets)
        resp = [None] * channels_num
        for i in range(channels_num):
            if self.channels_widgets[i].chosen.isChecked():
                resp[i] = self.channels_widgets[i].color
        self.coloring_update.emit(resp, new_image)

    def showEvent(self, event: QShowEvent):
        self.update_channels_list()
