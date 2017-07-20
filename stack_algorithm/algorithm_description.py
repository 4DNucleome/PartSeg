from qt_import import QDoubleSpinBox, QSpinBox, QComboBox, QWidget, QFormLayout, QAbstractSpinBox
import sys
from abc import ABCMeta, abstractmethod
from stack_settings import ImageSettings
from six import with_metaclass

if sys.version_info.major == 2:
    from exceptions import ValueError


class AlgorithmProperty(object):
    """
    :type name: str
    :type value_type: type
    :type default_value: object
    """
    def __init__(self, name, user_name, default_value, options_range, single_steep=None):
        self.name = name
        self.user_name = user_name
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
        return cls(name=ob.name, user_name=ob.user_name, default_value=ob.default_value, options_range=ob.range,
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
    def __init__(self, element_list, settings):
        """
        :type settings: ImageSettings
        :param element_list:
        :param settings:
        """
        super(AlgorithmSettingsWidget, self).__init__()
        self.widget_list = []
        widget_layout = QFormLayout()
        self.channels_chose = QComboBox()
        widget_layout.addRow("Channel", self.channels_chose)
        for el in element_list:
            self.widget_list.append((el.name, el.get_field()))
            widget_layout.addRow(el.user_name, self.widget_list[-1][-1])
        self.setLayout(widget_layout)
        self.settings = settings
        self.settings.image_changed[int].connect(self.image_changed)

    def image_changed(self, channels_num):
        ind = self.channels_chose.currentIndex()
        self.channels_chose.clear()
        self.channels_chose.addItems(map(str, range(channels_num)))
        self.channels_chose.setCurrentIndex(ind)

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

only_threshold_algorithm = [AlgorithmProperty("threshold", "Threshold", 1000, (0, 10**6), 100)]

threshold_algorithm = [AlgorithmProperty("threshold", "Threshold", 1000, (0, 10**6), 100),
                       AlgorithmProperty("minimum_size", "Minimum size", 80000, (0, 10**6), 1000)]

auto_threshold_algorithm = [AlgorithmProperty("suggested_size", "Suggested size", 80000, (0, 10**6), 1000),
                            AlgorithmProperty("threshold", "Minimum Threshold", 1000, (0, 10**6), 100),
                            AlgorithmProperty("minimum_size", "Minimum size", 40000, (0, 10**6), 1000)]


stack_algorithm_dict = {"Threshold": map(QtAlgorithmProperty.from_algorithm_property, threshold_algorithm),
                        "Auto Threshold": map(QtAlgorithmProperty.from_algorithm_property, auto_threshold_algorithm),
                        "Only Threshold": map(QtAlgorithmProperty.from_algorithm_property, only_threshold_algorithm)}