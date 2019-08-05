import math
import typing
from collections import defaultdict
from functools import partial

import numpy as np
from qtpy.QtCore import Signal, Qt, QRect, QRectF, QPointF, QSize, QModelIndex, QEvent, QPoint
from qtpy.QtGui import QImage, QShowEvent, QPaintEvent, QPainter, QPen, QMouseEvent, QColor, QPolygonF
from qtpy.QtWidgets import QWidget, QCheckBox, QGridLayout, QLabel, QHBoxLayout, QComboBox, QDoubleSpinBox, QListView, \
    QStyledItemDelegate, QStyleOptionViewItem, QStyle

from PartSeg.common_gui.numpy_qimage import create_colormap_image
from PartSeg.utils.color_image import ColorMap, create_color_map
from PartSeg.utils.color_image.base_colors import starting_colors
from .collapse_checkbox import CollapseCheckbox
from .universal_gui_part import CustomSpinBox
from ..common_backend.base_settings import ViewSettings
from ..utils.color_image import color_image

image_dict = {}  # dict to store QImages generated from colormap

ColorMapDict = typing.MutableMapping[str, typing.Tuple[ColorMap, bool]]


class ColorStyledDelegate(QStyledItemDelegate):
    """
    Class for paint :py:class:`~.ColorComboBox` elements when list trigger

    :param base_height: height of single list element
    :param color_dict: Dict mapping name to colors
    """
    def __init__(self, base_height: int, color_dict: ColorMapDict,  **kwargs):
        super().__init__(**kwargs)
        self.base_height = base_height
        self.color_dict = color_dict

    def paint(self, painter: QPainter, style: QStyleOptionViewItem, model: QModelIndex):
        if model.data() not in image_dict:
            image_dict[model.data()] = create_colormap_image(model.data(), self.color_dict)
        rect = QRect(style.rect.x(), style.rect.y() + 2, style.rect.width(), style.rect.height() - 4)
        painter.drawImage(rect, image_dict[model.data()])
        if int(style.state & QStyle.State_HasFocus):
            painter.save()
            pen = QPen()
            pen.setWidth(5)
            painter.setPen(pen)
            painter.drawRect(rect)
            painter.restore()

    def sizeHint(self, style: QStyleOptionViewItem, model: QModelIndex):
        res = super().sizeHint(style, model)
        # print(res)
        res.setHeight(self.base_height)
        res.setWidth(max(500, res.width()))
        return res


class ColorComboBox(QComboBox):
    """
    Combobox showing colormap instead of text

    :param id_num: id which be emit in signals. Designed to inform which channel information is changed
    :param colors: list of colors which should be able to chose. All needs to be keys in `color_dict`
    :param color_dict: dict from name to colormap definition
    :param colormap: initial colormap
    :param base_height: initial height of widgethow information that
    :param lock: show lock padlock to inform that fixed range is used
    :param blur: show info about blur selected
    """
    triangle_width = 20

    clicked = Signal(int)
    """Information about mouse click event on widget"""
    channel_visible_changed = Signal(int, bool)
    """Signal with information about change of channel visibility (ch_num, visible)"""
    channel_colormap_changed = Signal(int, str)
    """Signal with information about colormap change. (ch_num, name_of_colorma)"""

    def __init__(self, id_num: int, colors: typing.List[str], color_dict: ColorMapDict, colormap: str = "",
                 base_height=50, lock=False, blur=False):
        super().__init__()
        self.id = id_num
        self.check_box = QCheckBox()  # ColorCheckBox(parent=self)
        self.check_box.setChecked(True)
        self.lock = LockedInfoWidget(base_height - 10)
        self.lock.setVisible(lock)
        self.blur = BlurInfoWidget(base_height - 10)
        self.blur.setVisible(blur)
        self.color_dict = color_dict
        self.colors = colors
        self.addItems(self.colors)
        if colormap:
            self.color = colormap
        else:
            self.color = self.itemText(0)
        self.setCurrentText(self.color)
        self.currentTextChanged.connect(self._update_image)
        self.base_height = base_height
        self.show_arrow = False
        self.show_frame = False
        view = QListView()
        view.setMinimumWidth(200)
        view.setItemDelegate(ColorStyledDelegate(self.base_height, color_dict))
        self.setView(view)
        self.image = None  # only for moment, to reduce code repetition

        layout = QHBoxLayout()
        layout.setContentsMargins(7, 0, 0, 0)
        layout.addWidget(self.check_box)
        layout.addWidget(self.lock)
        layout.addWidget(self.blur)
        layout.addStretch(1)

        self.setLayout(layout)
        self.check_box.stateChanged.connect(partial(self.channel_visible_changed.emit, self.id))
        self.currentTextChanged.connect(partial(self.channel_colormap_changed.emit, self.id))
        self._update_image()

    def change_colors(self, colors: typing.List[str]):
        """change list of colormaps to choose"""
        self.colors = colors
        current_color = self.currentText()
        try:
            index = colors.index(current_color)
            prev = self.blockSignals(True)
        except ValueError:
            index = -1
            prev = self.signalsBlocked()

        self.clear()
        self.addItems(colors)
        if index != -1:
            self.setCurrentIndex(index)
            self.blockSignals(prev)
        else:
            self._update_image()
            self.repaint()

    def _update_image(self):
        self.color = self.currentText()
        if self.color not in image_dict:
            image_dict[self.color] = create_colormap_image(self.color, self.color_dict)
        self.image = image_dict[self.color]
        self.show_arrow = False

    def enterEvent(self, event: QEvent):
        self.show_arrow = True
        self.repaint()

    def mouseMoveEvent(self, _):
        self.show_arrow = True

    def leaveEvent(self, event: QEvent):
        self.show_arrow = False
        self.repaint()

    def showEvent(self, _event: QShowEvent):
        self.show_arrow = False

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.drawImage(self.rect(), self.image)
        if self.show_frame:
            painter.save()
            pen = QPen()
            pen.setWidth(2)
            pen.setColor(QColor("black"))
            painter.setPen(pen)
            rect = QRect(1, 1, self.width()-2, self.height()-2)
            painter.drawRect(rect)
            pen.setColor(QColor("white"))
            painter.setPen(pen)
            rect = QRect(3, 3, self.width() - 6, self.height() - 6)
            painter.drawRect(rect)
            painter.restore()
        if self.show_arrow:
            painter.save()
            triangle = QPolygonF()
            dist = 4
            point1 = QPoint(self.width() - self.triangle_width, 0)
            size = QSize(20, self.height() // 2)
            rect = QRect(point1, size)
            painter.fillRect(rect, QColor("white"))
            triangle.append(point1 + QPoint(dist, dist))
            triangle.append(point1 + QPoint(size.width() - dist, dist))
            triangle.append(point1 + QPoint(size.width() // 2, size.height() - dist))
            painter.setBrush(Qt.black)
            painter.drawPolygon(triangle, Qt.WindingFill)
            painter.restore()

    def mousePressEvent(self, event: QMouseEvent):
        if event.x() > self.width() - self.triangle_width:
            super().mousePressEvent(event)
        self.clicked.emit(self.id)

    def minimumSizeHint(self):
        size: QSize = super().minimumSizeHint()
        return QSize(size.width(), max(size.height(), self.base_height))

    def is_checked(self):
        """check if checkbox on widget is checked"""
        return self.check_box.isChecked()

    @property
    def is_lock(self):
        """check if lock property is set"""
        return self.lock.isVisible

    @property
    def is_blur(self):
        """check if blur property is set"""
        return self.blur.isVisible

    @property
    def set_lock(self):
        """set lock property"""
        return self.lock.setVisible

    @property
    def set_blur(self):
        """set blur property"""
        return self.blur.setVisible

    @property
    def colormap_changed(self):
        """alias for signal, return color name """
        return self.currentTextChanged

    def set_selection(self, val: bool):
        """set element selected (add frame)"""
        self.show_frame = val
        self.repaint()

    def set_color(self, val: str):
        """set current colormap"""
        self.setCurrentText(val)


class ChannelProperty(QWidget):
    """
    For manipulate chanel properties.
        1. Apply gaussian blur to channel
        2. Fixed range for coloring

    In future should be extended

    :param settings: for storing internal state. allow keep state between sessions
    :param start_name: name used to select proper information from settings object.
        Introduced for case with multiple image view.
    """
    def __init__(self, settings: ViewSettings, start_name: str):
        super().__init__()
        assert start_name != ""
        self.current_name = start_name
        self.current_channel = 0
        self._settings = settings
        self.widget_dict: typing.Dict[str, ColorComboBoxGroup] = {}

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

        self.collapse_widget = CollapseCheckbox("Channel property")
        self.collapse_widget.add_hide_element(self.minimum_value)
        self.collapse_widget.add_hide_element(self.maximum_value)
        self.collapse_widget.add_hide_element(self.fixed)
        self.collapse_widget.add_hide_element(self.use_gauss)
        self.collapse_widget.add_hide_element(self.gauss_radius)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.collapse_widget, 0, 0, 1, 4)
        label1 = QLabel("Min bright")
        layout.addWidget(label1, 1, 0)
        layout.addWidget(self.minimum_value, 1, 1)
        label2 = QLabel("Max bright")
        layout.addWidget(label2, 2, 0)
        layout.addWidget(self.maximum_value, 2, 1)
        layout.addWidget(self.fixed, 1, 2, 1, 2)
        layout.addWidget(self.use_gauss, 2, 2, 1, 1)
        layout.addWidget(self.gauss_radius, 2, 3, 1, 1)
        self.setLayout(layout)

        self.collapse_widget.add_hide_element(label1)
        self.collapse_widget.add_hide_element(label2)

    def send_info(self):
        """send info to """
        widget = self.widget_dict[self.current_name]
        widget.parameters_changed(self.current_channel)

    def register_widget(self, widget: 'ColorComboBoxGroup'):
        assert widget.name not in self.widget_dict
        self.widget_dict[widget.name] = widget
        self.change_current(widget.name, 0)

    def change_current(self, name, channel):
        assert name in self.widget_dict
        self.current_name = name
        self.current_channel = channel
        block = self.blockSignals(True)
        self.minimum_value.blockSignals(True)
        self.minimum_value.setValue(
            self._settings.get_from_profile(f"{self.current_name}.range_{self.current_channel}", (0, 65000))[0])
        self.minimum_value.blockSignals(False)
        self.maximum_value.setValue(
            self._settings.get_from_profile(f"{self.current_name}.range_{self.current_channel}", (0, 65000))[1])
        self.use_gauss.setChecked(
            self._settings.get_from_profile(f"{self.current_name}.use_gauss_{self.current_channel}", False))
        self.gauss_radius.setValue(
            self._settings.get_from_profile(f"{self.current_name}.gauss_radius_{self.current_channel}", 1))
        self.fixed.setChecked(
            self._settings.get_from_profile(f"{self.current_name}.lock_{self.current_channel}", False))
        self.blockSignals(block)

    def gauss_radius_changed(self):
        self._settings.set_in_profile(
            f"{self.current_name}.gauss_radius_{self.current_channel}", self.gauss_radius.value())
        if self.use_gauss.isChecked():
            self.send_info()

    def gauss_use_changed(self):
        self._settings.set_in_profile(
            f"{self.current_name}.use_gauss_{self.current_channel}", self.use_gauss.isChecked())
        self.send_info()

    def lock_channel(self, value):
        self._settings.set_in_profile(f"{self.current_name}.lock_{self.current_channel}", value)
        self.send_info()

    def range_changed(self):
        self._settings.set_in_profile(f"{self.current_name}.range_{self.current_channel}",
                                      (self.minimum_value.value(), self.maximum_value.value()))
        if self.fixed.isChecked():
            self.send_info()


class ColorComboBoxGroup(QWidget):
    """
    Group of :class:`.ColorComboBox` for control visibility and chose colormap for channels.
    """
    coloring_update = Signal()
    """information about global change of coloring"""
    change_channel = Signal([str, int])
    """information which channel change"""

    def __init__(self, settings: ViewSettings, name: str, channel_property: typing.Optional[ChannelProperty] = None,
                 height: int = 40):
        super().__init__()
        self.name = name
        self.height = height
        self.settings = settings
        self.settings.colormap_changes.connect(self.update_color_list)
        self.active_box = 0
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        if channel_property is not None:
            channel_property.register_widget(self)
            self.change_channel.connect(channel_property.change_current)

    def update_color_list(self, colors: typing.Optional[typing.List[str]] = None):
        """update list"""
        if colors is None:
            colors = self.settings.chosen_colormap
        for i in range(self.layout().count()):
            el: ColorComboBox = self.layout().itemAt(i).widget()
            el.change_colors(colors)

    def update_channels(self):
        """update number of channels base on settings"""
        self.set_channels(self.settings.channels)

    @property
    def channels_count(self):
        return self.layout().count()

    @property
    def current_colors(self) -> typing.List[typing.Optional[str]]:
        """"""
        resp = []
        for i in range(self.layout().count()):
            el: ColorComboBox = self.layout().itemAt(i).widget()
            if el.is_checked():
                resp.append(el.currentText())
            else:
                resp.append(None)
        return resp

    @property
    def current_colormaps(self):
        resp = []
        for i in range(self.layout().count()):
            el: ColorComboBox = self.layout().itemAt(i).widget()
            if el.is_checked():
                resp.append(self.settings.colormap_dict[el.currentText()][0])
            else:
                resp.append(None)
        return resp

    def change_selected_color(self, index, color):
        self.settings.set_channel_info(self.name, index, str(color))
        self.coloring_update.emit()
        self.change_channel.emit(self.name, index)

    def active_channel(self, chanel):
        return self.layout().itemAt(chanel).widget().is_checked()

    def set_channels(self, num: int):
        """Set number of channels to display"""
        self.settings.set_in_profile(f"{self.name}.channels_count", num)
        if num >= self.layout().count():
            for i in range(self.layout().count(), num):
                el = ColorComboBox(i, self.settings.chosen_colormap,
                                   self.settings.colormap_dict,
                                   self.settings.get_channel_info(self.name, i),
                                   base_height=self.height,
                                   lock=self.settings.get_from_profile(f"{self.name}.lock_{i}", False),
                                   blur=self.settings.get_from_profile(f"{self.name}.use_gauss_{i}", False)
                                   )
                el.clicked.connect(self.set_active)
                el.channel_visible_changed.connect(self.coloring_update)
                el.channel_colormap_changed.connect(self.change_selected_color)
                self.layout().addWidget(el)
        else:
            for i in range(self.layout().count() - num):
                el = self.layout().takeAt(num).widget()
                el.colormap_changed.disconnect()
                el.channel_colormap_changed.disconnect()
                el.clicked.disconnect()
                el.channel_visible_changed.disconnect()
                el.deleteLater()
        if num <= self.active_box:
            self.set_active(num - 1)
        self.change_channel.emit(self.name, self.active_box)

    def set_active(self, pos: int):
        self.active_box = pos
        for i in range(self.layout().count()):
            el = self.layout().itemAt(i).widget()
            if i == self.active_box:
                el.show_frame = True
            else:
                el.show_frame = False
        self.change_channel.emit(self.name, pos)
        self.repaint()

    def get_gauss(self):
        resp = []
        for i in range(self.layout().count()):
            resp.append((
                self.settings.get_from_profile(f"{self.name}.use_gauss_{i}", False),
                self.settings.get_from_profile(f"{self.name}.gauss_radius_{i}", 1)
            ))
        return resp

    def get_limits(self):
        resp: typing.List[typing.Union[typing.Tuple[int, int], None]] = [(0, 0)] * self.layout().count()  #
        for i in range(self.layout().count()):
            if not self.settings.get_from_profile(f"{self.name}.lock_{i}", False):
                resp[i] = None
            else:
                resp[i] = self.settings.get_from_profile(f"{self.name}.range_{i}", (0, 65000))
        return resp

    def parameters_changed(self, channel):
        """for ChannelProperty to inform about change of parameters"""
        if self.layout().itemAt(channel) is None:
            return
        widget: ColorComboBox = self.layout().itemAt(channel).widget()
        widget.set_blur(self.settings.get_from_profile(f"{self.name}.use_gauss_{channel}", False))
        widget.set_lock(self.settings.get_from_profile(f"{self.name}.lock_{channel}", False))
        if self.active_channel(channel):
            self.coloring_update.emit()
        if self.active_box == channel:
            self.change_channel.emit(self.name, channel)


class ColorPreview(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

    def paintEvent(self, event: QPaintEvent):
        rect = event.rect()
        painter = QPainter(self)
        if self.parent().image is not None:
            painter.drawImage(rect, self.parent().image)


class LockedInfoWidget(QWidget):
    """
    Widget used to present info about lock selection in class :py:class:`~.ColorComboBox`.
    """
    def __init__(self, size=25, margin=2):
        super().__init__()
        self.margin = margin
        self.setFixedWidth(size)
        self.setFixedHeight(size)

    def paintEvent(self, a0: QPaintEvent) -> None:
        super().paintEvent(a0)
        painter = QPainter(self)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        pen2 = QPen()

        rect = QRectF(self.margin, self.height() / 2, self.width() - self.margin * 2, self.height() / 2 - self.margin)
        rect2 = QRectF(3 * self.margin, 2 * self.margin, self.width() - self.margin * 6, self.height())

        pen2.setWidth(6)
        painter.setPen(pen2)
        painter.drawArc(rect2, 0, 180 * 16)
        pen2.setWidth(3)
        pen2.setColor(Qt.white)
        painter.setPen(pen2)
        painter.drawArc(rect2, 0, 180 * 16)

        painter.fillRect(rect, Qt.white)
        pen2.setWidth(2)
        pen2.setColor(Qt.black)
        painter.setPen(pen2)
        painter.drawRect(rect)

        painter.restore()


class BlurInfoWidget(QWidget):
    """
    Widget used to present info about blur selection in class :py:class:`~.ColorComboBox`.
    """
    def __init__(self, size=25, margin=2):
        super().__init__()
        self.margin = margin
        self.setFixedWidth(size)
        self.setFixedHeight(size)

    def paintEvent(self, a0: QPaintEvent) -> None:
        self.margin = 2

        super().paintEvent(a0)
        painter = QPainter(self)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(self.margin, self.margin, self.width() - self.margin * 2, self.height() - 2 * self.margin)
        painter.setBrush(Qt.white)
        painter.setPen(Qt.white)
        painter.drawEllipse(rect)

        painter.restore()
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen()
        pen.setWidth(2)
        painter.setPen(pen)
        mid_point = QPointF(a0.rect().width() / 2, a0.rect().height() / 2)
        radius = min(a0.rect().height(), a0.rect().width()) / 3
        rays_num = 10
        for i in range(rays_num):
            point = QPointF(math.sin(math.pi / (rays_num / 2) * i) * radius,
                            math.cos(math.pi / (rays_num / 2) * i) * radius)
            painter.drawLine(mid_point + (point * 0.4), mid_point + point)
        painter.restore()


class ChannelWidget(QWidget):
    clicked = Signal(int)

    def __init__(self, chanel_id: int, color: str, tight=False, locked=False, gauss=False):
        super().__init__()
        self.id = chanel_id
        self.active = False
        self.color = color
        self.chosen = QCheckBox(self)
        self.chosen.setChecked(True)
        self.chosen.setMinimumHeight(20)
        self.locked_info = LockedInfoWidget()
        self.locked_info.setVisible(locked)
        self.gauss_info = BlurInfoWidget()
        self.gauss_info.setVisible(gauss)
        layout = QHBoxLayout()
        layout.addWidget(self.chosen, 0)
        layout.addWidget(self.locked_info, 0)
        layout.addWidget(self.gauss_info, 0)
        layout.addStretch(1)
        if tight:
            layout.setContentsMargins(5, 0, 0, 0)
        else:
            layout.setContentsMargins(5, 1, 1, 1)
            self.setMinimumHeight(30)
        img = color_image(np.arange(0, 256).reshape((1, 256, 1)), [self.color], [(0, 256)])
        self.image = QImage(img.data, 256, 1, img.dtype.itemsize * 256 * 3, QImage.Format_RGB888)
        self.setLayout(layout)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if event.rect().top() == 0 and event.rect().left() == 0:
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
            scalar = image_length / length
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
        return self.locked_info.isVisible()

    def set_locked(self, val=True):
        if val:
            self.locked_info.setVisible(True)
        else:
            self.locked_info.setHidden(True)
        self.repaint()

    @property
    def gauss(self):
        return self.gauss_info.isVisible()

    def set_gauss(self, val=True):
        if val:
            self.gauss_info.setVisible(True)
        else:
            self.gauss_info.setHidden(True)
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


class MyComboBox(QComboBox):
    hide_popup = Signal()

    def hidePopup(self):
        super().hidePopup()
        self.hide_popup.emit()


class ChannelChooseBase(QWidget):
    coloring_update = Signal(bool)  # gave info if it is new image
    channel_change = Signal(int, bool)  # TODO something better for remove error during load different z-sizes images

    def __init__(self, settings: ViewSettings, parent=None, name="channelcontrol", text=""):
        super().__init__(parent)
        self._settings = settings
        self._name = name
        self._main_name = name
        self.text = text
        self.image = None

        self.channels_widgets = []
        self.channels_layout = QHBoxLayout()
        self.channels_layout.setContentsMargins(0, 0, 0, 0)

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
    # TODO I add some possibility to synchronization state between two image views. Im not sure it is well projected
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
        self.channels_widgets[self.current_channel].set_gauss(self.use_gauss.isChecked())
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
        self._settings.set_channel_info(self.name, self.current_channel, str(value))
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
            self.channels_widgets.append(
                ChannelWidget(i, self._settings.get_channel_info(
                              self.name, i, starting_colors[i % len(starting_colors)]),
                              locked=self._settings.get_from_profile(f"{self._name}.lock_{i}", False),
                              gauss=self._settings.get_from_profile(f"{self._name}.use_gauss_{i}", False)))
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
        self.fixed.setChecked(self._settings.get_from_profile(f"{self._name}.lock_{index}", False))
        self.current_channel = 0
        self.image = self.channels_widgets[0].image
        self.colormap_chose.setCurrentText(self.channels_widgets[0].color)
        self.send_info(True)
        self.channel_change.emit(self.current_channel, True)

    def get_current_colors(self):
        channels_num = len(self.channels_widgets)
        resp = []
        for i in range(channels_num):
            resp.append(self._settings.get_channel_info(self.name, i,  self.channels_widgets[i].color))
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
        resp: typing.List[typing.Union[typing.Tuple[int, int], None]] = [(0, 0)] * channels_num  #
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
        pass  # self.update_channels_list()
