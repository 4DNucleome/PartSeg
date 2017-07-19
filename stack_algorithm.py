from qt_import import QDoubleSpinBox, QSpinBox, QComboBox, QWidget, QFormLayout, QAbstractSpinBox
import sys
from abc import ABCMeta, abstractmethod
from six import with_metaclass

if sys.version_info.major == 2:
    from exceptions import ValueError


class AlgorithmProperty(object):
    """
    :type name: str
    :type value_type: type
    :type default_value: object
    """
    def __init__(self, name, default_value, options_range, single_steep=None):
        self.name = name
        self.value_type = type(default_value)
        self.default_value = default_value
        self.range = options_range
        self.single_step = single_steep
        if self.value_type is list:
            assert default_value in options_range


class QtAlgorithmProperty(AlgorithmProperty):
    qt_class_dict = {int: QSpinBox, float: QDoubleSpinBox, list: QComboBox}

    def __init__(self, *args, **kwargs):
        super(QtAlgorithmProperty, self).__init__(*args, **kwargs)

    @classmethod
    def from_algorithm_property(cls, ob):
        """
        :type ob: AlgorithmProperty
        :param ob: AlgorithmProperty object
        :return: QtAlgorithmProperty
        """
        return cls(name=ob.name, default_value=ob.default_value, options_range=ob.range,
                   single_steep=ob.single_step)

    def get_field(self):
        field = self.qt_class_dict[self.value_type]()
        if isinstance(field, QComboBox):
            field.addItems(self.range)
            field.setCurrentIndex(self.range.index(self.default_value))
        else:
            field.setRange(*self.range)
            field.setValue(self.default_value)
            if self.single_step is not None:
                field.setSingleStep(self.single_step)
        return field


class AbstractAlgorithmSettingsWidget(with_metaclass(ABCMeta, object)):
    def __init__(self):
        pass

    @abstractmethod
    def get_values(self):
        """
        :return: dict[str, object]
        """
        return dict()


class AlgorithmSettingsWidget(QWidget):
    def __init__(self, element_list):
        super(AlgorithmSettingsWidget, self).__init__()
        self.widget_list = []
        widget_layout = QFormLayout()
        for el in element_list:
            self.widget_list.append((el.name, el.get_field()))
            widget_layout.addRow(*self.widget_list[-1])
        self.setLayout(widget_layout)

    def get_values(self):
        res = dict()
        for name, el in self.widget_list:
            if isinstance(el, QComboBox):
                res[name] = str(el.currentText())
            elif isinstance(el, QAbstractSpinBox):
                res[name] = el.value()
            else:
                raise ValueError("unsuported type {}".format(type(el)))
        return res

AbstractAlgorithmSettingsWidget.register(AlgorithmSettingsWidget)


threshold_algorithm = [AlgorithmProperty("Threshold", 1000, (0, 10**6), 100),
                       AlgorithmProperty("Minimum size", 80000, (0, 10**6), 1000)]

auto_threshold_algorithm = [AlgorithmProperty("Suggested size", 80000, (0, 10**6), 1000),
                            AlgorithmProperty("Minimum Threshold", 1000, (0, 10**6), 100),
                            AlgorithmProperty("Minimum size", 40000, (0, 10**6), 1000)]


stack_algorithm_dict = {"Threshold": map(QtAlgorithmProperty.from_algorithm_property, threshold_algorithm),
                        "Auto Threshold": map(QtAlgorithmProperty.from_algorithm_property, auto_threshold_algorithm)}