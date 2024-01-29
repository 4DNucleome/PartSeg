"""
This module contains simple, useful widgets which implementation is too short to create separated files for them
"""

import math
import typing
import warnings
from enum import Enum

from magicgui import register_type
from magicgui.widgets import Combobox
from qtpy import PYSIDE2
from qtpy.QtCore import QPointF, QRect, Qt, QTimer
from qtpy.QtGui import QColor, QFontMetrics, QPainter, QPaintEvent, QPalette, QPolygonF
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QTextEdit,
    QWidget,
)
from superqt import QEnumComboBox

from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import Channel

enum_type = object if PYSIDE2 else Enum


class ChannelComboBox(QComboBox):
    """Combobox for selecting channel index. Channel numeration starts from 1 for user and from 0 for developer"""

    def get_value(self) -> Channel:
        """Return current channel. Starting from 0"""
        return Channel(self.currentIndex())

    def set_value(self, val: typing.Union[Channel, int]):
        """Set current channel . Starting from 0"""
        if isinstance(val, Channel):
            self.setCurrentIndex(val.value)
            return
        self.setCurrentIndex(val)

    def change_channels_num(self, num: int):
        """Change number of channels"""
        block = self.blockSignals(True)
        index = self.currentIndex()
        self.clear()
        self.addItems(list(map(str, range(1, num + 1))))
        if index < 0 or index > num:
            index = 0
        self.blockSignals(block)
        self.setCurrentIndex(index)


class MguiChannelComboBox(Combobox):
    """Combobox for selecting channel index. Channel numeration starts from 1 for user and from 0 for developer"""

    def change_channels_num(self, num: int):
        """Change number of channels"""
        self.choices = [Channel(i) for i in range(num)]  # pylint: disable=attribute-defined-outside-init
        # TODO understand why pylint do not se choices property in magicgui

    def __init__(self, **kwargs):
        super().__init__(choices=[Channel(i) for i in range(10)], **kwargs)


register_type(Channel, widget_type=MguiChannelComboBox)


EnumType = typing.TypeVar("EnumType", bound=Enum)


class EnumComboBox(QEnumComboBox):
    """
    Combobox for choose :py:class:`enum.Enum` values

    :param enum: Enum on which base combo box should be created.
        For proper showing labels overload the ``__str__`` function of given :py:class:`enum.Enum`
    """

    def __init__(self, enum, parent=None):
        warnings.warn(
            "EnumComboBox is deprecated, use superqt.QEnumComboBox instead", category=DeprecationWarning, stacklevel=2
        )
        super().__init__(parent=parent, enum_class=enum)

    def get_value(self) -> EnumType:
        """current value as Enum member"""
        return self.currentEnum()

    @property
    def current_choose(self):
        """current value as Enum member"""
        return self.currentEnumChanged

    def _emit_signal(self):
        self.current_choose.emit(self.get_value())

    def set_value(self, value: typing.Union[EnumType, int]):
        """Set value with Eunum or int"""
        if isinstance(value, int):
            self.setCurrentIndex(value)
        else:
            self.setCurrentEnum(value)


class Spacing(QWidget):
    """
    :type elements: list[QDoubleSpinBox | QSpinBox]
    """

    def __init__(
        self,
        title: str,
        data_sequence: typing.Sequence[typing.Union[float, int]],
        unit: Units,
        parent=None,
        input_type: QAbstractSpinBox = QDoubleSpinBox,
        data_range: typing.Tuple[float, float] = (0, 100000),
    ):
        """
        :param title: title of the widget
        :param data_sequence: initial values of the widget
        :param unit: unit of the values
        :param parent: parent widget
        :param input_type: type of the input widget
        :type data_range: (float, float)
        """
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.addWidget(QLabel(f"<strong>{title}</strong>"))
        self.elements = []
        if len(data_sequence) == 2:
            data_sequence = (1, *tuple(data_sequence))
        for name, value in zip(["z", "y", "x"], data_sequence):
            lab = QLabel(f"{name}:")
            layout.addWidget(lab)
            val = QDoubleSpinBox()
            val.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            val.setRange(*data_range)
            val.setValue(value * UNIT_SCALE[unit.value])
            val.setAlignment(Qt.AlignmentFlag.AlignRight)
            font = val.font()
            fm = QFontMetrics(font)
            # TODO check width attribute
            val_len = max(fm.width(str(data_range[0])), fm.width(str(data_range[1]))) + fm.width(" " * 8)
            val.setFixedWidth(val_len)
            layout.addWidget(val)
            self.elements.append(val)
        self.units = QEnumComboBox(enum_class=Units)
        self.units.setCurrentEnum(unit)
        layout.addWidget(self.units)
        self.has_units = True
        layout.addStretch(1)
        self.setLayout(layout)

    def get_values(self):
        return [x.value() / UNIT_SCALE[self.units.currentEnum().value] for x in self.elements]

    def set_values(self, value_list):
        for val, wid in zip(value_list, self.elements):
            wid.setValue(val)

    def get_unit_str(self):
        return self.units.currentText() if self.has_units else ""


def right_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
    return label


class CustomSpinBox(QSpinBox):
    """
    Spin box for integer with dynamic single steep

    :param bounds: Bounds for changing single step. Default value:
        ``((300, 1), (1000, 10), (10000, 100)), 1000``
        Format:
        ``(List[(threshold, single_step)], default_single_step)``
        the single_step is chosen by checking upper bound of threshold of
        current spin box value
    """

    def __init__(self, *args, bounds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        if bounds is not None:
            warnings.warn("bounds parameter is deprecated", FutureWarning, stacklevel=2)  # pragma: no cover


class CustomDoubleSpinBox(QDoubleSpinBox):
    """
    Spin box for float with dynamic single steep

    :param bounds: Bounds for changing single step. Default value:
        ``((0.2, 0.01), (2, 0.1), (300, 1), (1000, 10), (10000, 100)), 1000``
        Format:
        ``(List[(threshold, single_step)], default_single_step)``
        the single_step is chosen by checking upper bound of threshold of
        current spin box value
    """

    def __init__(self, *args, bounds=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStepType(QAbstractSpinBox.StepType.AdaptiveDecimalStepType)
        if bounds is not None:
            warnings.warn("bounds parameter is deprecated", FutureWarning, stacklevel=2)  # pragma: no cover


class ProgressCircle(QWidget):
    """
    This is widget for generating circuital progress bar

    :param background: color of background circle, need to be acceptable by :py:class:`PyQt5.QtGui.QColor` constructor.
    :param main_color: color of progress marking, need to be acceptable by :py:class:`PyQt5.QtGui.QColor` constructor.

    .. warning::
      This widget currently have no minimum size. You need to specify it in your code
    """

    def __init__(self, background: QColor = "white", main_color: QColor = "darkCyan", parent=None):
        super().__init__(parent)
        self.nominator = 1
        self.denominator = 1
        self.background_color = QColor(background)
        self.main_color = QColor(main_color)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        size = min(self.width(), self.height())
        rect = QRect(0, 0, size, size)
        painter.setBrush(self.background_color)
        painter.setPen(self.background_color)
        painter.drawEllipse(rect)
        painter.setBrush(self.main_color)
        painter.setPen(self.main_color)
        factor = self.nominator / self.denominator
        radius = size / 2
        if factor > 0.5:
            painter.drawChord(rect, 0, int(16 * 360 * 0.5))
            painter.drawChord(rect, 16 * 180, int(16 * 360 * (factor - 0.5)))
            zero_point = QPointF(0, radius)
        else:
            painter.drawChord(rect, 0, int(16 * 360 * factor))
            zero_point = QPointF(size, radius)
        mid_point = QPointF(radius, radius)
        point = mid_point + QPointF(
            math.cos(math.pi * (factor * 2)) * radius, -math.sin(math.pi * (factor * 2)) * radius
        )
        polygon = QPolygonF([mid_point, zero_point, point])
        painter.drawPolygon(polygon)
        painter.restore()

    def set_fraction(self, nominator, denominator=1):
        """
        Set fraction.

        :param nominator: of fraction if denominator is set to 1 then nominator should be from range [0, 1]
        :param denominator: as name
        """
        self.nominator = nominator
        self.denominator = denominator
        self.repaint()


class InfoLabel(QWidget):
    """
    Label for cyclic showing text from list. It uses :py:class:`~.ProgressCircle`
    to inform user abu time to change text.

    :param text_list: texts to be in cyclic use
    :param delay: time in milliseconds between changes
    :param parent: passed to :py:class:`QWidget` constructor
    """

    def __init__(self, text_list: typing.List[str], delay: int = 10000, parent=None):
        if len(text_list) == 0:
            raise ValueError("List of text to show should be non empty.")
        super().__init__(parent)
        self.text_list = text_list
        self.index = 0
        self.delay = delay
        self.time = 0
        self.time_step = 100
        self.timer = QTimer()
        self.timer.setSingleShot(False)
        self.timer.timeout.connect(self.one_step)
        self.timer.setInterval(self.time_step)
        self.setToolTip("Double click for change suggestion. Hold mouse over for stop changes.")
        self.label = QLabel()
        self.progress = ProgressCircle()
        self.progress.setFixedHeight(25)
        self.progress.setFixedWidth(25)
        layout = QHBoxLayout()
        layout.addWidget(self.progress)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.change_text()

    def mouseDoubleClickEvent(self, _):
        """Force change text"""
        self.change_text()
        self.time = 0
        self.progress.set_fraction(self.time, self.delay)

    def enterEvent(self, _):
        """stop timer"""
        self.timer.stop()

    def leaveEvent(self, _):
        """Start timer"""
        self.timer.start()

    def showEvent(self, _):
        """Start timer"""
        self.timer.start()

    def hideEvent(self, _):
        """Stop timer"""
        self.timer.stop()

    def one_step(self):
        self.time += self.time_step
        if self.time > self.delay:
            self.time = 0
            self.change_text()
        self.progress.set_fraction(self.time, self.delay)

    def change_text(self):
        """Change text in cyclic mode"""
        self.label.setText(self.text_list[self.index])
        self.index = (self.index + 1) % len(self.text_list)


class TextShow(QTextEdit):
    """
    Show text with word wrap and scroll if needed.
    Limit to show first :py:attr:`lines` without scroll bar.
    """

    def __init__(self, lines=5, text="", parent=None):
        super().__init__(text, parent)
        self.lines = lines
        self.setReadOnly(True)
        p: QPalette = self.palette()
        p.setColor(QPalette.ColorRole.Base, p.color(self.backgroundRole()))
        self.setPalette(p)

    def height(self):
        metrics = QFontMetrics(self.currentFont())
        height = metrics.height()
        return height * 5

    def sizeHint(self):
        s = super().sizeHint()
        metrics = QFontMetrics(self.currentFont())
        height = metrics.height()
        s.setHeight(int(height * (self.lines + 0.5)))
        return s


class Hline(QWidget):
    """Horizontal line"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(3)

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.drawLine(0, 0, self.width(), 0)
