import typing
from abc import ABCMeta, abstractmethod
from copy import deepcopy
from enum import Enum
from typing import Type, List

from PyQt5.QtGui import QHideEvent, QPaintEvent, QPainter
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QFormLayout, \
    QAbstractSpinBox, QScrollArea, QLineEdit, QHBoxLayout, QStackedWidget, QGridLayout
from PyQt5.QtCore import pyqtSignal

from six import with_metaclass

from common_gui.dim_combobox import DimComboBox
from common_gui.universal_gui_part import CustomSpinBox, CustomDoubleSpinBox
from project_utils.channel_class import Channel
from project_utils.segmentation.algorithm_base import SegmentationAlgorithm
from project_utils.error_dialog import ErrorDialog
from project_utils.image_operations import to_radius_type_dict, RadiusType
from project_utils.segmentation.algorithm_describe_base import AlgorithmProperty, AlgorithmDescribeBase
from project_utils.segmentation_thread import SegmentationThread
from project_utils.universal_const import UNIT_SCALE
from tiff_image import Image

from project_utils.settings import ImageSettings, BaseSettings


class EnumComboBox(QComboBox):
    def __init__(self, enum: type(Enum), parent=None):
        super().__init__(parent=parent)
        self.enum = enum

    def get_value(self):
        return list(self.enum.__members__.values())[self.currentIndex()]

    def set_value(self, value: Enum):
        self.setCurrentText(value.name)


class ChannelComboBox(QComboBox):
    def get_value(self):
        return self.currentIndex()

    def set_value(self, val):
        self.setCurrentText(str(val))

    def change_channels_num(self, num):
        index = self.currentIndex()
        self.clear()
        self.addItems(map(str, range(num)))
        if index < 0 or index > num:
            index = 0
        self.setCurrentIndex(index)


class QtAlgorithmProperty(AlgorithmProperty):
    qt_class_dict = {int: CustomSpinBox, float: CustomDoubleSpinBox, list: QComboBox, bool: QCheckBox,
                     RadiusType: DimComboBox}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widget = self._get_field()
        self.change_fun = self.get_change_signal(self._widget)
        self._getter, self._setter = self.get_setter_and_getter_function(self._widget)

    def get_value(self):
        return self._getter(self._widget)

    def set_value(self, val):
        return self._setter(self._widget, val)

    def get_field(self):
        return self._widget

    @classmethod
    def from_algorithm_property(cls, ob):
        """
        :type ob: AlgorithmProperty | str
        :param ob: AlgorithmProperty object or label
        :return: QtAlgorithmProperty | QLabel
        """
        if isinstance(ob, AlgorithmProperty):
            return cls(name=ob.name, user_name=ob.user_name, default_value=ob.default_value, options_range=ob.range,
                       single_steep=ob.single_step, property_type=ob.value_type, possible_values=ob.possible_values)
        elif isinstance(ob, str):
            return QLabel(ob)
        raise ValueError(f"unknown parameter type {type(ob)} of {ob}")

    def _get_field(self) -> QWidget:
        if issubclass(self.value_type, Channel):
            res = ChannelComboBox()
            res.addItems([str(x) for x in range(10)])
            return res
        elif issubclass(self.value_type, AlgorithmDescribeBase):
            res = SubAlgorithmWidget(self)
        elif issubclass(self.value_type, bool):
            res = QCheckBox()
            res.setChecked(bool(self.default_value))
        elif issubclass(self.value_type, int):
            res = CustomSpinBox()
            assert isinstance(self.default_value, int)
            res.setValue(self.default_value)
            if self.range is not None:
                res.setRange(*self.range)
        elif issubclass(self.value_type, float):
            res = CustomDoubleSpinBox()
            assert isinstance(self.default_value, float)
            res.setValue(self.default_value)
            if self.range is not None:
                res.setRange(*self.range)
        elif issubclass(self.value_type, str):
            res = QLineEdit()
            res.setText(str(self.default_value))
        elif issubclass(self.value_type, Enum):
            res = EnumComboBox(self.value_type)
            # noinspection PyUnresolvedReference,PyUnresolvedReferences
            res.addItems(self.value_type.__members__.keys())
            # noinspection PyUnresolvedReferences
            res.set_value(self.default_value)
        else:
            raise ValueError(f"Unknown class: {self.value_type}")
        return res

    @staticmethod
    def get_change_signal(widget: QWidget):
        if isinstance(widget, QComboBox):
            return widget.currentIndexChanged
        elif isinstance(widget, QCheckBox):
            return widget.stateChanged
        elif isinstance(widget, (CustomDoubleSpinBox, CustomSpinBox)):
            return widget.valueChanged
        elif isinstance(widget, QLineEdit):
            return widget.textChanged
        elif isinstance(widget, SubAlgorithmWidget):
            return widget.values_changed
        raise ValueError(f"Unsupported type: {type(widget)}")

    @staticmethod
    def get_setter_and_getter_function(widget: QWidget):
        if isinstance(widget, ChannelComboBox):
            return widget.__class__.get_value, widget.__class__.set_value
        if isinstance(widget, EnumComboBox):
            return widget.__class__.get_value, widget.__class__.set_value
        if isinstance(widget, QComboBox):
            return widget.__class__.currentText, widget.__class__.setCurrentText
        elif isinstance(widget, QCheckBox):
            return widget.__class__.isChecked, widget.__class__.setChecked
        elif isinstance(widget, CustomSpinBox):
            return widget.__class__.value, widget.__class__.setValue
        elif isinstance(widget, CustomDoubleSpinBox):
            return widget.__class__.value, widget.__class__.setValue
        elif isinstance(widget, QLineEdit):
            return widget.__class__.text, widget.__class__.setText
        elif isinstance(widget, SubAlgorithmWidget):
            return widget.__class__.get_values, widget.__class__.set_values
        raise ValueError(f"Unsupported type: {type(widget)}")


class FormWidget(QWidget):
    value_changed = pyqtSignal()

    def __init__(self, fields: typing.List[AlgorithmProperty]):
        super().__init__()
        self.widgets_dict: typing.Dict[str, QtAlgorithmProperty] = dict()
        self.channels_chose: typing.List[typing.Union[ChannelComboBox, SubAlgorithmWidget]] = []
        layout = QFormLayout()
        element_list = map(QtAlgorithmProperty.from_algorithm_property, fields)
        for el in element_list:
            if isinstance(el, QLabel):
                layout.addRow(el)
            elif isinstance(el.get_field(), SubAlgorithmWidget):
                layout.addRow(QLabel(el.user_name), el.get_field().choose)
                layout.addRow(el.get_field())
                self.widgets_dict[el.name] = el
            else:
                self.widgets_dict[el.name] = el
                layout.addRow(QLabel(el.user_name), el.get_field())
                # noinspection PyUnresolvedReferences
                el.change_fun.connect(self.value_changed)
                if issubclass(el.value_type, Channel):
                    self.channels_chose.append(el.get_field())
        """row = 0
        for el in element_list:
            if isinstance(el, QLabel):
                layout.addWidget(el, row, 0, 1, 2)
            elif isinstance(el.get_field(), SubAlgorithmWidget):
                layout.addWidget(QLabel(el.user_name), row, 0)
                layout.addWidget(el.get_field().choose, row, 1)
                layout.addWidget(el.get_field(), row + 1, 0, 1, 2)
                row += 1
            else:
                self.widgets_dict[el.name] = el
                layout.addWidget(QLabel(el.user_name), row, 0)
                layout.addWidget(el.get_field(), row, 1)
                # noinspection PyUnresolvedReferences
                el.change_fun.connect(self.value_changed)
                if issubclass(el.value_type, Channel):
                    self.channels_chose.append(el.get_field())
            row += 1
            """

        self.setLayout(layout)

    def get_values(self):
        return dict(((name, el.get_value()) for name, el in self.widgets_dict.items()))

    def set_values(self, values: dict):
        for name, value in values.items():
            if name in self.widgets_dict:
                self.widgets_dict[name].set_value(value)

    def image_changed(self, image: Image):
        for channel_widget in self.channels_chose:
            if isinstance(channel_widget, ChannelComboBox):
                channel_widget.change_channels_num(image.channels)
            else:
                channel_widget.change_channels_num(image)


class SubAlgorithmWidget(QWidget):
    values_changed = pyqtSignal()

    def __init__(self, property: AlgorithmProperty):
        super().__init__()
        print(property)
        assert isinstance(property.possible_values, dict)
        assert isinstance(property.default_value, str)
        self.property = property
        self.widget_dict: typing.Dict[str, FormWidget] = {}
        # TODO protect for recursion
        widget = FormWidget(property.possible_values[property.default_value].get_fields())
        widget.layout().setContentsMargins(0,0,0,0)

        self.widget_dict[property.default_value] = widget
        self.choose = QComboBox(self)
        self.choose.addItems(property.possible_values.keys())
        self.setContentsMargins(0, 0, 0, 0)

        self.choose.setCurrentText(property.default_value)

        self.choose.currentTextChanged.connect(self.algorithm_choose)
        # self.setStyleSheet("border: 1px solid red")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(widget)
        tmp_widget = QWidget(self)
        tmp_widget.setMinimumHeight(5000)
        layout.addWidget(tmp_widget)
        self.tmp_widget = tmp_widget
        self.setLayout(layout)

    """def add_to_layout(self, layout: QFormLayout):
        print("[add_to_layout]")
        lay1 = self.layout().takeAt(0).layout()
        label = lay1.takeAt(0).widget()
        lay1.removeWidget(label)
        lay1.removeWidget(self.choose)
        self.layout().removeWidget(self.current_widget)
        layout.addRow(label.text(), self.choose)
        layout.addRow(self.current_widget)
        self.current_layout = layout"""

    def set_values(self, val: dict):
        self.choose.setCurrentText(val["name"])
        if val["name"] not in self.widget_dict:
            self.algorithm_choose(val["name"])
        self.widget_dict[val["name"]].set_values(val["values"])

    def get_values(self):
        name = self.choose.currentText()
        values = self.widget_dict[name].get_values()
        return {"name": name, "values": values}


    def change_channels_num(self, image: Image):
        for i in range(self.layout().count()):
            el = self.layout().itemAt(i)
            if el.widget() and isinstance(el.widget(), FormWidget):
                el.widget().image_changed(image)

    def algorithm_choose(self, name):
        print(f"change to name: {name}")
        if name not in self.widget_dict:
            self.widget_dict[name] = FormWidget(self.property.possible_values[name].get_fields())
            print(self.widget_dict[name].minimumSize())
            self.widget_dict[name].layout().setContentsMargins(0, 0, 0, 0)
            self.layout().addWidget(self.widget_dict[name])
        widget = self.widget_dict[name]
        for i in range(self.layout().count()):
            lay_elem = self.layout().itemAt(i)
            if lay_elem.widget():
                lay_elem.widget().hide()
        widget.show()

    def showEvent(self, _event):
        # workaround for changing size
        self.tmp_widget.hide()


class AbstractAlgorithmSettingsWidget(with_metaclass(ABCMeta, object)):
    def __init__(self):
        pass

    @abstractmethod
    def get_values(self):
        """
        :return: dict[str, object]
        """
        return dict()


class BaseAlgorithmSettingsWidget(QScrollArea):
    algorithm_thread: SegmentationThread
    gauss_radius_name = "gauss_radius"
    use_gauss_name = "use_gauss"

    def __init__(self, settings: BaseSettings, name, algorithm: Type[SegmentationAlgorithm]):
        """
        For algorithm which works on one channel
        :type settings: ImageSettings
        :param settings:
        """
        super().__init__()
        self.widget_list = []
        self.name = name
        self.algorithm = algorithm
        main_layout = QVBoxLayout()
        self.info_label = QLabel()
        self.info_label.setHidden(True)
        main_layout.addWidget(self.info_label)
        self.form_widget = FormWidget(algorithm.get_fields())
        self.setWidget(self.form_widget)
        self.settings = settings
        value_dict = self.settings.get(f"algorithms.{self.name}", {})
        self.set_values(value_dict)
        self.settings.image_changed[Image].connect(self.image_changed)
        self.algorithm_thread = SegmentationThread(algorithm())
        self.algorithm_thread.info_signal.connect(self.show_info)
        self.algorithm_thread.exception_occurred.connect(self.exception_occurred)

    def exception_occurred(self, exc: Exception):
        dial = ErrorDialog(exc, "Error during segmentation", f"{self.name}")
        dial.exec()

    def show_info(self, text):
        self.info_label.setText(text)
        self.info_label.setVisible(True)

    def image_changed(self, image: Image):
        self.algorithm_thread.algorithm.set_image(image)
        self.form_widget.image_changed(image)

    def set_values(self, values_dict):
        self.form_widget.set_values(values_dict)

    def get_values(self):
        return self.form_widget.get_values()

    def channel_num(self):
        return self.channels_chose.currentIndex()

    def execute(self, exclude_mask=None):
        values = self.get_values()
        self.settings.set(f"algorithms.{self.name}", deepcopy(values))
        scale = UNIT_SCALE[self.settings.get("units_index")]
        print(f"values: {values}")
        self.algorithm_thread.set_parameters(**values)
        self.algorithm_thread.start()

    def hideEvent(self, a0: QHideEvent):
        self.algorithm_thread.clean()


class AlgorithmSettingsWidget(BaseAlgorithmSettingsWidget):
    def execute(self, exclude_mask=None):
        self.algorithm_thread.algorithm.set_exclude_mask(exclude_mask)
        self.algorithm_thread.algorithm.set_image(self.settings.image)
        super().execute(exclude_mask)


class InteractiveAlgorithmSettingsWidget(BaseAlgorithmSettingsWidget):
    algorithm_thread: SegmentationThread

    def __init__(self, settings, name, algorithm: Type[SegmentationAlgorithm],
                 selector: List[QWidget]):
        super().__init__(settings, name, algorithm)
        self.selector = selector
        self.algorithm_thread.finished.connect(self.enable_selector)
        self.algorithm_thread.started.connect(self.disable_selector)
        self.form_widget.value_changed.connect(self.value_updated)
        # noinspection PyUnresolvedReferences
        settings.mask_changed.connect(self.change_mask)

    def value_updated(self):
        if not self.parent().interactive:
            return
        self.execute()

    def change_mask(self):
        if not self.isVisible():
            return
        self.algorithm_thread.algorithm.set_mask(self.settings.mask)

    def disable_selector(self):
        for el in self.selector:
            el.setDisabled(True)

    def enable_selector(self):
        for el in self.selector:
            el.setEnabled(True)


AbstractAlgorithmSettingsWidget.register(AlgorithmSettingsWidget)
