from PyQt5.QtWidgets import QWidget, QSpinBox, QCheckBox, QGridLayout, QLabel, QHBoxLayout, QComboBox, QDoubleSpinBox
from PyQt5.QtGui import QImage, QShowEvent, QPaintEvent, QPainter, QPen, QMouseEvent
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
from common_gui.collapse_checkbox import CollapseCheckbox
from project_utils.color_image import color_image
import typing

from stackseg.stack_settings import default_colors
from project_utils.settings import ViewSettings

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
        self.info_label = QLabel("\U0001F512")
        self.info_label.setHidden(True)
        self.info_label.setStyleSheet("QLabel {background-color: white; border-radius: 3px;} ")
        layout = QHBoxLayout()
        layout.addWidget(self.chosen, 0)
        layout.addWidget(self.info_label, 0)
        layout.addStretch(1)
        #layout.addWidget(QLabel("aa"), 1)
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
            pen1.setWidth(0)
            painter.setPen(pen1)
        super().paintEvent(event)

    def set_active(self, val=True):
        self.active = val
        self.repaint()

    @property
    def locked(self):
        return self.info_label.isVisible()

    def set_locked(self, val=True):
        if val:
            self.info_label.setVisible(True)
        else:
            self.info_label.setHidden(True)
        self.repaint()

    def set_inactive(self, val=True):
        self.active = not val
        self.repaint()

    def set_color(self, color):
        self.color = color
        img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [self.color], [(0, 255)])
        self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
        self.repaint()

    def mousePressEvent(self, a0: QMouseEvent):
        self.clicked.emit(self.id)
        print(self.color)


class MyComboBox(QComboBox):
    hide_popup = pyqtSignal()

    def hidePopup(self):
        super().hidePopup()
        self.hide_popup.emit()


class CustomSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.valueChanged.connect(self.value_changed)

    def value_changed(self, val: int):
        if val < 300:
            self.setSingleStep(1)
        elif val < 1000:
            self.setSingleStep(10)
        elif val < 10000:
            self.setSingleStep(100)
        else:
            self.setSingleStep(1000)



class ChannelControl(QWidget):
    #TODO improve adding own scaling
    """
    :type channels_widgets: typing.List[ChannelWidget]
    :type _settings: .stack_settings.ImageSettings
    """

    coloring_update = pyqtSignal(bool)

    def __init__(self, settings : ViewSettings, parent=None, flags=Qt.WindowFlags(), name="channelcontrol"):
        super().__init__(parent, flags)
        self._name = name
        self._settings = settings
        self.current_channel = 0
        self.current_bounds = settings.get_from_profile(f"{self._name}.bounds", [])
        self.colormap_chose = MyComboBox()
        self.colormap_chose.addItems(self._settings.chosen_colormap)
        self.colormap_chose.highlighted[str].connect(self.change_color_preview)
        self.colormap_chose.hide_popup.connect(self.change_closed)
        self.colormap_chose.activated[str].connect(self.change_color)
        self.channel_preview_widget = ColorPreview(self)
        # self.channel_preview_widget.setPixmap(QPixmap.fromImage(self.channel_preview))
        self.minimum_value = CustomSpinBox(self)
        self.minimum_value.setRange(-10**6, 10**6)
        self.minimum_value.valueChanged.connect(self.range_changed)
        self.maximum_value = CustomSpinBox(self)
        self.maximum_value.setRange(-10 ** 6, 10 ** 6)
        self.maximum_value.valueChanged.connect(self.range_changed)
        self.fixed = QCheckBox("Fix range")
        self.fixed.stateChanged.connect(self.lock_channel)
        self.gauss = QCheckBox("Gauss")
        self.gauss.setToolTip("Only current channel")
        self.gauss.stateChanged.connect(self.coloring_update.emit)
        self.gauss_radius = QDoubleSpinBox()
        self.collapse_widget = CollapseCheckbox()
        self.collapse_widget.add_hide_element(self.minimum_value)
        self.collapse_widget.add_hide_element(self.maximum_value)
        self.collapse_widget.add_hide_element(self.fixed)
        self.collapse_widget.add_hide_element(self.gauss)
        self.collapse_widget.add_hide_element(self.gauss_radius)

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
        layout.addWidget(label2, 5, 0)
        layout.addWidget(self.maximum_value, 5, 1)
        layout.addWidget(self.fixed, 4, 2, 1, 2)
        layout.addWidget(self.gauss, 5, 2, 1, 1)
        layout.addWidget(self.gauss_radius, 5, 3, 1, 1)
        self.collapse_widget.add_hide_element(label1)
        self.collapse_widget.add_hide_element(label2)
        self.setLayout(layout)
        self.collapse_widget.setChecked(True)
        self._settings.image_changed.connect(self.update_channels_list)

    def lock_channel(self, value):
        self.channels_widgets[self.current_channel].set_locked(value)
        self.coloring_update.emit(False)

    def range_changed(self):
        self.current_bounds[self.current_channel] = self.minimum_value.value(), self.maximum_value.value()
        self._settings.set_in_profile(f"{self._name}.bounds", self.current_bounds)
        if self.fixed.isChecked():
            self.coloring_update.emit(False)

    def change_chanel(self, id):
        if id == self.current_channel:
            return
        self.minimum_value.setValue(self.current_bounds[id][0])
        self.maximum_value.setValue(self.current_bounds[id][1])
        self.channels_widgets[self.current_channel].set_inactive()
        self.current_channel = id

        self.channels_widgets[id].set_active()
        self.fixed.setChecked(self.channels_widgets[id].locked)
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
        self.channels_widgets[self.current_channel].set_color(value)
        self.change_color_preview(value)
        self._settings.set_in_profile(f"{self._name}.cmap{self.current_channel}", str(value))
        self.send_info()


    def update_channels_list(self):
        channels_num = self._settings.channels
        for i in range(len(self.current_bounds), channels_num):
            self.current_bounds.append([0, 65000])
        self._settings.set_in_profile(f"{self._name}.bounds", self.current_bounds)
        for el in self.channels_widgets:
            self.channels_layout.removeWidget(el)
            el.clicked.disconnect()
            el.chosen.stateChanged.disconnect()
            el.deleteLater()
        self.channels_widgets = []
        for i in range(channels_num):
            self.channels_widgets.append(ChannelWidget(i, self._settings.get_from_profile(f"{self._name}.cmap{i}",
                                                       default_colors[i % len(default_colors)])))
            self.channels_layout.addWidget(self.channels_widgets[-1])
            self.channels_widgets[-1].clicked.connect(self.change_chanel)
            self.channels_widgets[-1].chosen.stateChanged.connect(self.send_info_wrap)
        self.channels_widgets[0].set_active()
        self.minimum_value.setValue(self.current_bounds[0][0])
        self.maximum_value.setValue(self.current_bounds[0][1])
        self.current_channel = 0
        self.image = self.channels_widgets[0].image
        self.colormap_chose.setCurrentText(self.channels_widgets[0].color)
        self.send_info(True)

    def send_info_wrap(self):
        self.send_info()

    @property
    def current_colors(self):
        channels_num = len(self.channels_widgets)
        resp :typing.List[typing.Union[str, None]] = [None] * channels_num
        for i in range(channels_num):
            if self.channels_widgets[i].chosen.isChecked():
                resp[i] = self.channels_widgets[i].color
        return resp

    def get_limits(self):
        channels_num = len(self.channels_widgets)
        resp: typing.List[typing.Union[typing.Tuple[int, int], None]] = self.current_bounds[:channels_num]
        for i in range(channels_num):
            if not self.channels_widgets[i].locked:
                resp[i] = None
        return resp

    def active_cannel(self, index):
        return self.channels_widgets[index].chosen.isChecked()

    def send_info(self, new_image=False):
        self.coloring_update.emit(new_image)

    def range_update(self):
        self.coloring_update.emit(False)

    def showEvent(self, event: QShowEvent):
        self.update_channels_list()
