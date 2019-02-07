# coding=utf-8
from sys import platform
from enum import Enum
from qtpy.QtCore import Qt
from qtpy.QtGui import QFontMetrics

from PartSeg.utils.universal_const import Units, UNIT_SCALE
from .flow_layout import FlowLayout
from qtpy.QtWidgets import QWidget, QLabel, QDoubleSpinBox, QAbstractSpinBox, QSpinBox, QComboBox, QSlider,\
    QLineEdit, QHBoxLayout


class ChannelComboBox(QComboBox):
    def get_value(self):
        return self.currentIndex()

    def set_value(self, val):
        self.setCurrentIndex(int(val))

    def change_channels_num(self, num):
        index = self.currentIndex()
        self.clear()
        self.addItems(map(str, range(1, num + 1)))
        if index < 0 or index > num:
            index = 0
        self.setCurrentIndex(index)


class EnumComboBox(QComboBox):
    def __init__(self, enum: type(Enum), parent=None):
        super().__init__(parent=parent)
        self.enum = enum
        self.addItems(list(map(str,  enum.__members__.values())))

    def get_value(self) -> Enum:
        return list(self.enum.__members__.values())[self.currentIndex()]

    def set_value(self, value: Enum):
        self.setCurrentText(value.name)



class Spacing(QWidget):
    """
    :type elements: list[QDoubleSpinBox | QSpinBox]
    """
    def __init__(self, title, data_sequence, unit: Units, parent=None, input_type=QDoubleSpinBox, decimals=2, data_range=(0, 100000),
                 single_step=1):
        """
        :type data_sequence: list[(float)]
        :param data_sequence:
        :type input_type: () -> (QDoubleSpinBox | QSpinBox)
        :param parent:
        :type decimals: int|None
        :type data_range: (float, float)
        :type single_step: float
        :type title: str
        """
        super().__init__(parent)
        layout = QHBoxLayout()
        layout.addWidget(QLabel("<strong>{}</strong>".format(title)))
        self.elements = []
        if len(data_sequence) == 2:
            data_sequence = (1,) + tuple(data_sequence)
        for name, value in zip(["z", "y", "x"], data_sequence):
            lab = QLabel(name+":")
            layout.addWidget(lab)
            val = input_type()
            val.setButtonSymbols(QAbstractSpinBox.NoButtons)
            if isinstance(val, QDoubleSpinBox):
                val.setDecimals(decimals)
            val.setRange(*data_range)
            val.setValue(value * UNIT_SCALE[unit.value])
            val.setAlignment(Qt.AlignRight)
            val.setSingleStep(single_step)
            font = val.font()
            fm = QFontMetrics(font)
            val_len = max(fm.width(str(data_range[0])), fm.width(str(data_range[1]))) + fm.width(" "*8)
            val.setFixedWidth(val_len)
            layout.addWidget(val)
            self.elements.append(val)
        self.units = EnumComboBox(Units)
        self.units.set_value(unit)
        layout.addWidget(self.units)
        self.has_units = True
        #layout.addStretch()
        layout.addStretch(1)
        self.setLayout(layout)

    def get_values(self):
        return [x.value() / UNIT_SCALE[self.units.get_value().value] for x in self.elements]

    def set_values(self, value_list):
        for val, wid in zip(value_list, self.elements):
            wid.setValue(val)

    def get_unit_str(self):
        if self.has_units:
            return self.units.currentText()
        else:
            return ""


def right_label(text):
    label = QLabel(text)
    label.setAlignment(Qt.AlignVCenter | Qt.AlignRight)
    return label


def set_position(elem, previous, dist=10):
    pos_y = previous.pos().y()
    if platform.system() == "Darwin" and isinstance(elem, QLineEdit):
        pos_y += 3
    if platform.system() == "Darwin" and isinstance(previous, QLineEdit):
        pos_y -= 3
    if platform.system() == "Darwin" and isinstance(previous, QSlider):
        pos_y -= 10
    if platform.system() == "Darwin" and isinstance(elem, QSpinBox):
        pos_y += 7
    if platform.system() == "Darwin" and isinstance(previous, QSpinBox):
        pos_y -= 7
    elem.move(previous.pos().x() + previous.size().width() + dist, pos_y)


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


class CustomDoubleSpinBox(QDoubleSpinBox):
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