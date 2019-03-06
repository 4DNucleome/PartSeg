from collections import defaultdict

from qtpy.QtWidgets import QWidget, QCheckBox, QGridLayout, QLabel, QHBoxLayout, QComboBox, QDoubleSpinBox
from qtpy.QtGui import QImage, QShowEvent, QPaintEvent, QPainter, QPen, QMouseEvent
from qtpy.QtCore import Signal
import numpy as np
import typing
from .collapse_checkbox import CollapseCheckbox
from .universal_gui_part import CustomSpinBox
from ..utils.color_image import color_image
from ..project_utils_qt.settings import ViewSettings


default_colors = ['BlackRed', 'BlackGreen', 'BlackBlue', 'BlackMagenta']


class ColorPreview(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        if self.parent().image is not None:
            painter.drawImage(rect, self.parent().image)


class ChannelWidget(QWidget):
    clicked = Signal(int)

    def __init__(self, chanel_id: int, color: str, tight=False):
        super().__init__()
        self.id = chanel_id
        self.active = False
        self.color = color
        self.chosen = QCheckBox(self)
        self.chosen.setChecked(True)
        self.chosen.setMinimumHeight(20)
        self.info_label = QLabel("<small>\U0001F512</small>")
        self.info_label.setHidden(True)
        self.info_label.setStyleSheet("QLabel {background-color: white; border-radius: 3px; margin: 0px; padding: 0px} ")
        self.info_label.setMargin(0)
        layout = QHBoxLayout()
        layout.addWidget(self.chosen, 0)
        layout.addWidget(self.info_label, 0)
        layout.addStretch(1)
        if tight:
            layout.setContentsMargins(5,0,0,0)
        else:
            layout.setContentsMargins(5,1, 1, 1)
            self.setMinimumHeight(30)
        # layout.addWidget(QLabel("aa"), 1)
        img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [self.color], [(0, 256)])
        self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
        self.setLayout(layout)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if event.rect().top()  == 0 and event.rect().left() == 0:
            rect = event.rect()

            painter.drawImage(rect, self.image)
            if self.active:
                pen = QPen()
                pen.setWidth(5)
                pen1 = painter.pen()
                painter.setPen(pen)
                painter.drawRect(event.rect())
                pen1.setWidth(0)
                painter.setPen(pen1)
        else:
            length = self.rect().width()
            image_length = self.image.width()
            scalar = image_length/length
            begin = int(event.rect().x() * scalar)
            width = int(event.rect().width() * scalar)
            image_cut = self.image.copy(begin, 0, width, 1)
            painter.drawImage(event.rect(), image_cut)
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
        #print(self.color)


class MyComboBox(QComboBox):
    hide_popup = Signal()

    def hidePopup(self):
        super().hidePopup()
        self.hide_popup.emit()


class ChannelChooseBase(QWidget):
    coloring_update = Signal(bool)  # gave info if it is new image
    channel_change = Signal(int, bool) # TODO something better for remove error during load different z-sizes images

    def __init__(self, settings: ViewSettings, parent=None, name="channelcontrol", text=""):
        super().__init__(parent)
        self._settings = settings
        self._name = name
        self._main_name = name
        self.text = text
        self.image = None


        self.channels_widgets = []
        self.channels_layout = QHBoxLayout()
        self.channels_layout.setContentsMargins(0,0,0,0)

    def lock_channel(self, value):
        self.channels_widgets[self.current_channel].set_locked(value)
        self._settings.set_in_profile(f"{self._name}.lock_{self.current_channel}", value)
        self.coloring_update.emit(False)
        self.channel_change.emit(self.current_channel, False)

    def send_info_wrap(self):
        self.send_info()

    def _set_layout(self):
        raise NotImplementedError()

    def send_info(self, new_image=False):
        self.coloring_update.emit(new_image)

    def get_limits(self):
        raise NotImplementedError()

    def get_gauss(self):
        raise NotImplementedError()


class ChannelControl(ChannelChooseBase):
    # TODO improve adding own scaling
    # TODO I add some possibiity to synchronization state beeten two image wiews. Im not shure it is well projected
    """
    :type channels_widgets: typing.List[ChannelWidget]
    """

    parameters_changed = Signal()

    # noinspection PyUnresolvedReferences
    def __init__(self, settings: ViewSettings, parent=None, name="channelcontrol", text=""):
        super().__init__(settings, parent, name, text)

        self._settings.colormap_changes.connect(self.colormap_list_changed)

        self.enable_synchronize = False
        self.current_channel = 0
        # self.current_bounds = settings.get_from_profile(f"{self._name}.bounds", [])
        self.colormap_chose = MyComboBox()
        self.colormap_chose.addItems(self._settings.chosen_colormap)
        self.colormap_chose.highlighted[str].connect(self.change_color_preview)
        self.colormap_chose.hide_popup.connect(self.change_closed)
        self.colormap_chose.activated[str].connect(self.change_color)
        self.channel_preview_widget = ColorPreview(self)
        # self.channel_preview_widget.setPixmap(QPixmap.fromImage(self.channel_preview))
        self.minimum_value = CustomSpinBox(self)
        self.minimum_value.setRange(-10 ** 6, 10 ** 6)
        self.minimum_value.valueChanged.connect(self.range_changed)
        self.maximum_value = CustomSpinBox(self)
        self.maximum_value.setRange(-10 ** 6, 10 ** 6)
        self.maximum_value.valueChanged.connect(self.range_changed)
        self.fixed = QCheckBox("Fix range")
        self.fixed.stateChanged.connect(self.lock_channel)
        self.use_gauss = QCheckBox("Gauss")
        self.use_gauss.setToolTip("Only current channel")
        self.gauss_radius = QDoubleSpinBox()
        self.gauss_radius.setSingleStep(0.1)
        self.gauss_radius.valueChanged.connect(self.gauss_radius_changed)
        self.use_gauss.stateChanged.connect(self.gauss_use_changed)
        self.collapse_widget = CollapseCheckbox()
        self.collapse_widget.add_hide_element(self.minimum_value)
        self.collapse_widget.add_hide_element(self.maximum_value)
        self.collapse_widget.add_hide_element(self.fixed)
        self.collapse_widget.add_hide_element(self.use_gauss)
        self.collapse_widget.add_hide_element(self.gauss_radius)
        self.setContentsMargins(0, 0, 0, 0)

        # layout.setVerticalSpacing(0)
        self._set_layout()
        self.collapse_widget.setChecked(True)
        self._settings.image_changed.connect(self.update_channels_list)
        self.minimum_value.setMinimumWidth(50)
        self.maximum_value.setMinimumWidth(50)
        self.gauss_radius.setMinimumWidth(30)

    def _set_layout(self):
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        if self.text != "":
            layout.addWidget(QLabel(self.text), 0, 0, 1, 4)
        layout.addLayout(self.channels_layout, 1, 0, 1, 4)
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
        layout.addWidget(self.use_gauss, 5, 2, 1, 1)
        layout.addWidget(self.gauss_radius, 5, 3, 1, 1)
        self.setLayout(layout)

        self.collapse_widget.add_hide_element(label1)
        self.collapse_widget.add_hide_element(label2)

    @property
    def name(self):
        return self._name

    def colormap_list_changed(self):
        self.colormap_chose.blockSignals(True)
        text = self.colormap_chose.currentText()
        self.colormap_chose.clear()
        colormaps: list = self._settings.chosen_colormap
        self.colormap_chose.addItems(colormaps)
        try:
            index = colormaps.index(text)
            self.colormap_chose.setCurrentIndex(index)
        except KeyError:
            pass
        self.colormap_chose.blockSignals(False)

    def refresh_info(self):
        """Function for synchronization preview settings between two previews"""
        self.coloring_update.emit(False)

    def set_temp_name(self, new_name=None):
        """Function for synchronization preview settings between two previews"""
        if new_name is None:
            self._name = self._main_name
        else:
            self._name = new_name
        self.coloring_update.emit(False)

    def range_changed(self):
        self._settings.set_in_profile(f"{self._name}.range_{self.current_channel}",
                                      (self.minimum_value.value(), self.maximum_value.value()))
        # self.current_bounds[self.current_channel] = self.minimum_value.value(), self.maximum_value.value()
        # self._settings.set_in_profile(f"{self._name}.bounds", self.current_bounds)
        if self.fixed.isChecked():
            self.coloring_update.emit(False)
            self.channel_change.emit(self.current_channel, False)

    def gauss_radius_changed(self):
        self._settings.set_in_profile(f"{self._name}.gauss_radius_{self.current_channel}", self.gauss_radius.value())
        if self.use_gauss.isChecked():
            self.coloring_update.emit(False)

    def gauss_use_changed(self):
        self._settings.set_in_profile(f"{self._name}.use_gauss_{self.current_channel}", self.use_gauss.isChecked())
        self.coloring_update.emit(False)

    def change_chanel(self, chanel_id):
        if chanel_id == self.current_channel:
            return
        self.channels_widgets[self.current_channel].set_inactive()
        self.current_channel = chanel_id
        self.minimum_value.blockSignals(True)
        self.minimum_value.setValue(self._settings.get_from_profile(f"{self._name}.range_{chanel_id}", (0, 65000))[0])
        self.minimum_value.blockSignals(False)
        self.maximum_value.setValue(self._settings.get_from_profile(f"{self._name}.range_{chanel_id}", (0, 65000))[1])
        self.use_gauss.setChecked(self._settings.get_from_profile(f"{self._name}.use_gauss_{chanel_id}", False))
        self.gauss_radius.setValue(self._settings.get_from_profile(f"{self._name}.gauss_radius_{chanel_id}", 1))

        self.channels_widgets[chanel_id].set_active()
        self.fixed.setChecked(self.channels_widgets[chanel_id].locked)
        self.image = self.channels_widgets[chanel_id].image
        self.colormap_chose.setCurrentText(self.channels_widgets[chanel_id].color)
        self.channel_preview_widget.repaint()
        self.channel_change.emit(chanel_id, False)

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
        self.channel_change.emit(self.current_channel, False)
        self.send_info()

    def update_channels_list(self):
        """update list of channels on image change"""
        channels_num = self._settings.channels
        index = 0
        is_checked = defaultdict(lambda: True)
        for i, el in enumerate(self.channels_widgets):
            self.channels_layout.removeWidget(el)
            if el.active:
                index = i
            is_checked[i] = el.chosen.isChecked()
            el.clicked.disconnect()
            el.chosen.stateChanged.disconnect()
            el.deleteLater()
        self.channels_widgets = []
        for i in range(channels_num):
            self.channels_widgets.append(ChannelWidget(i, self._settings.get_from_profile(f"{self._name}.cmap{i}",
                                                                                          default_colors[i % len(
                                                                                              default_colors)])))
            self.channels_widgets[-1].chosen.setChecked(is_checked[i])
            self.channels_layout.addWidget(self.channels_widgets[-1])
            self.channels_widgets[-1].clicked.connect(self.change_chanel)
            self.channels_widgets[-1].chosen.stateChanged.connect(self.send_info_wrap)
        if index >= len(self.channels_widgets):
            index = 0
        self.channels_widgets[index].set_active()
        self.minimum_value.blockSignals(True)
        self.minimum_value.setValue(self._settings.get_from_profile(f"{self._name}.range_{index}", (0, 65000))[0])
        self.minimum_value.blockSignals(False)
        self.maximum_value.setValue(self._settings.get_from_profile(f"{self._name}.range_{index}", (0, 65000))[1])
        self.use_gauss.setChecked(self._settings.get_from_profile(f"{self._name}.use_gauss_{index}", False))
        self.gauss_radius.setValue(self._settings.get_from_profile(f"{self._name}.gauss_radius_{index}", 1))
        self.current_channel = 0
        self.image = self.channels_widgets[0].image
        self.colormap_chose.setCurrentText(self.channels_widgets[0].color)
        self.send_info(True)
        self.channel_change.emit(self.current_channel, True)

    def get_current_colors(self):
        channels_num = len(self.channels_widgets)
        resp = []
        for i in range(channels_num):
            resp.append(self._settings.get_from_profile(f"{self._name}.cmap{i}", self.channels_widgets[i].color))
        return resp

    @property
    def current_colors(self):
        channels_num = len(self.channels_widgets)
        resp = self.get_current_colors()
        for i in range(channels_num):
            if not self.channels_widgets[i].chosen.isChecked():
                resp[i] = None
        return resp

    def get_limits(self):
        channels_num = len(self.channels_widgets)
        resp = [(0, 0)] * channels_num
        # : typing.List[typing.Union[typing.Tuple[int, int], None]]
        for i in range(channels_num):
            if not self._settings.get_from_profile(f"{self._name}.lock_{i}", False):
                resp[i] = None
            else:
                resp[i] = self._settings.get_from_profile(f"{self._name}.range_{i}", (0, 65000))
        return resp

    def get_gauss(self):
        channels_num = len(self.channels_widgets)
        resp = []
        for i in range(channels_num):
            resp.append((
                self._settings.get_from_profile(f"{self._name}.use_gauss_{i}", False),
                self._settings.get_from_profile(f"{self._name}.gauss_radius_{i}", 1)
            ))
        return resp

    def active_channel(self, index):
        return self.channels_widgets[index].chosen.isChecked()

    def range_update(self):
        self.coloring_update.emit(False)

    def showEvent(self, event: QShowEvent):
        pass # self.update_channels_list()


class ChannelChoose(ChannelChooseBase):
    """
    Only chose which channels are visible
    """
    def __init__(self, settings: ViewSettings, main_channel_control: ChannelControl, parent=None,
                 name="channelchoose", text=""):
        super().__init__(settings, parent, name, text)
        self.main_channel_control = main_channel_control
        self._settings.image_changed.connect(self.update_channels_list)
        self.main_channel_control.channel_change.connect(self.color_change)
        self.main_channel_control.coloring_update.connect(self.coloring_update_resend)
        self._set_layout()

    def coloring_update_resend(self, val):
        self.coloring_update.emit(val)

    def color_change(self, val, new=False):
        if val >= len(self.channels_widgets):
            return
        colormap = self._settings.get_from_profile(f"{self.main_channel_control.name}.cmap{val}")
        self.channels_widgets[val].set_color(colormap)
        value = self._settings.get_from_profile(f"{self.name}.lock_{val}")
        self.channels_widgets[val].set_locked(value)
        self.send_info(new)
        self.channel_change.emit(val, False)

    def get_limits(self):
        return self.main_channel_control.get_limits()

    def get_gauss(self):
        return self.main_channel_control.get_gauss()

    @property
    def current_colors(self):
        channels_num = len(self.channels_widgets)
        resp = self.main_channel_control.get_current_colors()
        for i in range(channels_num):
            if not self.channels_widgets[i].chosen.isChecked():
                resp[i] = None
        return resp

    def _set_layout(self):
        self.setLayout(self.channels_layout)

    def update_channels_list(self):
        channels_num = self._settings.channels
        is_checked = defaultdict(lambda: True)
        for i, el in enumerate(self.channels_widgets):
            self.channels_layout.removeWidget(el)
            is_checked[i] = el.chosen.isChecked()
            el.clicked.disconnect()
            el.chosen.stateChanged.disconnect()
            el.deleteLater()
        self.channels_widgets = []
        for i in range(channels_num):
            self.channels_widgets.append(ChannelWidget(
                i, self._settings.get_from_profile(f"{self.main_channel_control.name}.cmap{i}",
                                                   default_colors[i % len(default_colors)]), True))
            self.channels_widgets[-1].chosen.setChecked(is_checked[i])
            self.channels_layout.addWidget(self.channels_widgets[-1])
            self.channels_widgets[-1].clicked.connect(self.change_chanel)
            self.channels_widgets[-1].chosen.stateChanged.connect(self.send_info_wrap)
        self.channels_widgets[0].set_active()
        self.image = self.channels_widgets[0].image

        self.current_channel = 0
        self.send_info(True)
        self.channel_change.emit(self.current_channel, True)

    def change_chanel(self, chanel_id):
        if chanel_id == self.current_channel:
            return
        self.channels_widgets[self.current_channel].set_inactive()
        self.current_channel = chanel_id
        self.channels_widgets[chanel_id].set_active()
        self.image = self.channels_widgets[chanel_id].image
        self.channel_change.emit(chanel_id, False)


    def active_channel(self, index):
        return self.channels_widgets[index].chosen.isChecked()

    @property
    def name(self):
        return self.main_channel_control.name
