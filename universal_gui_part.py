# coding=utf-8
from qt_import import QWidget, QHBoxLayout, QLabel, Qt, QDoubleSpinBox, QAbstractSpinBox, QSpinBox, QComboBox


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
        layout = QHBoxLayout()
        layout.addWidget(QLabel("<strong>{}</strong>".format(title)))
        self.elements = []
        print(data_sequence)
        for name, value in data_sequence:
            lab = right_label(name)
            layout.addWidget(lab)
            val = input_type()
            val.setButtonSymbols(QAbstractSpinBox.NoButtons)
            val.setValue(value)
            if isinstance(val, QDoubleSpinBox):
                val.setDecimals(decimals)
            val.setRange(*data_range)
            val.setAlignment(Qt.AlignRight)
            val.setSingleStep(single_step)
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
        layout.addStretch()
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
