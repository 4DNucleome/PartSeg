# coding=utf-8
from sys import platform

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFontMetrics

from .flow_layout import FlowLayout
from PyQt5.QtWidgets import QWidget, QLabel, QDoubleSpinBox, QAbstractSpinBox, QSpinBox, QComboBox, QSlider,\
    QLineEdit


class Spacing(QWidget):
    """
    :type elements: list[QDoubleSpinBox | QSpinBox]
    """
    def __init__(self, title, data_sequence, parent=None, input_type=QDoubleSpinBox, decimals=2, data_range=(0, 1000),
                 single_step=1, units=None, units_index=0):
        """
        :type data_sequence: list[(str, float)]
        :param data_sequence:
        :type input_type: () -> (QDoubleSpinBox | QSpinBox)
        :param parent:
        :type decimals: int|None
        :type data_range: (float, float)
        :type single_step: float
        :type title: str
        :type units: None|list[str]
        :type units_index: int
        """
        super(Spacing, self).__init__(parent)
        layout = FlowLayout()
        layout.addWidget(QLabel("<strong>{}</strong>".format(title)))
        self.elements = []
        print(data_sequence)
        for name, value in data_sequence:
            lab = right_label(name)
            layout.addWidget(lab)
            val = input_type()
            val.setButtonSymbols(QAbstractSpinBox.NoButtons)
            if isinstance(val, QDoubleSpinBox):
                val.setDecimals(decimals)
            val.setRange(*data_range)
            val.setValue(value)
            val.setAlignment(Qt.AlignRight)
            val.setSingleStep(single_step)
            font = val.font()
            fm = QFontMetrics(font)
            val_len = max(fm.width(str(data_range[0])), fm.width(str(data_range[1]))) + fm.width(" "*8)
            val.setFixedWidth(val_len)
            layout.addWidget(val)
            self.elements.append(val)
        if units is not None:
            self.units = QComboBox()
            self.units.addItems(units)
            self.units.setCurrentIndex(units_index)
            layout.addWidget(self.units)
            self.has_units = True
        else:
            self.has_units = False
        #layout.addStretch()
        self.setLayout(layout)

    def get_values(self):
        return [x.value() for x in self.elements]

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