import collections
import typing
from copy import deepcopy
from enum import Enum

from qtpy.QtCore import Signal
from qtpy.QtGui import QHideEvent, QPainter, QPaintEvent, QResizeEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from PartSeg.common_gui.error_report import ErrorDialog
from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, ROIExtractionProfile
from PartSegCore.channel_class import Channel
from PartSegCore.image_operations import RadiusType
from PartSegCore.segmentation.algorithm_base import (
    SegmentationAlgorithm,
    SegmentationLimitException,
    SegmentationResult,
)
from PartSegImage import Image

from ..common_backend.base_settings import BaseSettings
from ..common_backend.segmentation_thread import SegmentationThread
from .dim_combobox import DimComboBox
from .universal_gui_part import ChannelComboBox, CustomDoubleSpinBox, CustomSpinBox, EnumComboBox


def update(d, u):
    if not isinstance(d, dict):
        d = {}
    for k, v in u.items():
        d[k] = update(d.get(k, {}), v) if isinstance(v, collections.abc.Mapping) else v
    return d


def _pretty_print(data, indent=2) -> str:
    if isinstance(data, dict):
        res = "\n"
        for k, v in data.items():
            res += f"{' ' * indent}{k}: {_pretty_print(v, indent+2)}\n"
        return res[:-1]
    return str(data)


class QtAlgorithmProperty(AlgorithmProperty):
    qt_class_dict = {
        int: CustomSpinBox,
        float: CustomDoubleSpinBox,
        list: QComboBox,
        bool: QCheckBox,
        RadiusType: DimComboBox,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widget = self._get_field()
        self.change_fun = self.get_change_signal(self._widget)
        self._getter, self._setter = self.get_getter_and_setter_function(self._widget)
        self._setter(self._widget, self.default_value)

    def get_value(self):
        return self._getter(self._widget)

    def recursive_get_values(self):
        if isinstance(self._widget, SubAlgorithmWidget):
            return self._widget.recursive_get_values()
        return self.get_value()

    def set_value(self, val):
        """set value of widget """
        try:
            return self._setter(self._widget, val)
        except TypeError:
            pass

    def get_field(self) -> QWidget:
        """
        Get representing widget
        :return:
        :rtype:
        """
        return self._widget

    @classmethod
    def from_algorithm_property(cls, ob):
        """
        Create class instance base on :py:class:`.AlgorithmProperty` instance

        :type ob: AlgorithmProperty | str
        :param ob: AlgorithmProperty object or label
        :return: QtAlgorithmProperty | QLabel
        """
        if isinstance(ob, AlgorithmProperty):
            return cls(
                name=ob.name,
                user_name=ob.user_name,
                default_value=ob.default_value,
                options_range=ob.range,
                single_steep=ob.single_step,
                value_type=ob.value_type,
                possible_values=ob.possible_values,
                help_text=ob.help_text,
                per_dimension=ob.per_dimension,
            )
        if isinstance(ob, str):
            return QLabel(ob)
        raise ValueError(f"unknown parameter type {type(ob)} of {ob}")

    def _get_field(self) -> QWidget:
        """
        Get proper widget for given field type. Overwrite if would like to support new data types.
        """
        if self.per_dimension:
            self.per_dimension = False
            prop = self.from_algorithm_property(self)
            self.per_dimension = True
            res = ListInput(prop, 3)
        elif issubclass(self.value_type, Channel):
            res = ChannelComboBox()
            res.change_channels_num(10)
            return res
        elif issubclass(self.value_type, AlgorithmDescribeBase):
            res = SubAlgorithmWidget(self)
        elif issubclass(self.value_type, bool):
            res = QCheckBox()
        elif issubclass(self.value_type, int):
            res = CustomSpinBox()
            if not isinstance(self.default_value, int):
                raise ValueError(
                    f"Incompatible types. default_value should be type of int. Is {type(self.default_value)}"
                )
            if self.range is not None:
                res.setRange(*self.range)
        elif issubclass(self.value_type, float):
            res = CustomDoubleSpinBox()
            if not isinstance(self.default_value, float):
                raise ValueError(
                    f"Incompatible types. default_value should be type of float. Is {type(self.default_value)}"
                )
            if self.range is not None:
                res.setRange(*self.range)
        elif issubclass(self.value_type, str):
            res = QLineEdit()
        elif issubclass(self.value_type, Enum):
            res = EnumComboBox(self.value_type)
            # noinspection PyUnresolvedReferences
        elif issubclass(self.value_type, list):
            res = QComboBox()
            res.addItems(list(map(str, self.possible_values)))
        elif hasattr(self.value_type, "get_object"):
            res = self.value_type.get_object()
        else:
            raise ValueError(f"Unknown class: {self.value_type}")
        tool_tip_text = ""
        if self.help_text:
            tool_tip_text = self.help_text
        tool_tip_text += f"default value: {_pretty_print(self.default_value)}"
        res.setToolTip(tool_tip_text)
        return res

    @staticmethod
    def get_change_signal(widget: QWidget):
        if isinstance(widget, QComboBox):
            return widget.currentIndexChanged
        if isinstance(widget, QCheckBox):
            return widget.stateChanged
        if isinstance(widget, (CustomDoubleSpinBox, CustomSpinBox)):
            return widget.valueChanged
        if isinstance(widget, QLineEdit):
            return widget.textChanged
        if isinstance(widget, SubAlgorithmWidget):
            return widget.values_changed
        if isinstance(widget, ListInput):
            return widget.change_signal
        if hasattr(widget, "values_changed"):
            return widget.values_changed
        raise ValueError(f"Unsupported type: {type(widget)}")

    @staticmethod
    def get_getter_and_setter_function(
        widget: QWidget,
    ) -> typing.Tuple[
        typing.Callable[
            [
                QWidget,
            ],
            typing.Any,
        ],
        typing.Callable[[QWidget, typing.Any], None],  # noqa E231
    ]:
        """
        For each widget type return proper functions. This functions need instance as first argument

        :return: (getter, setter)
        """
        if isinstance(widget, ChannelComboBox):
            return widget.__class__.get_value, widget.__class__.set_value
        if isinstance(widget, EnumComboBox):
            return widget.__class__.get_value, widget.__class__.set_value
        if isinstance(widget, QComboBox):
            return widget.__class__.currentText, widget.__class__.setCurrentText
        if isinstance(widget, QCheckBox):
            return widget.__class__.isChecked, widget.__class__.setChecked
        if isinstance(widget, CustomSpinBox):
            return widget.__class__.value, widget.__class__.setValue
        if isinstance(widget, CustomDoubleSpinBox):
            return widget.__class__.value, widget.__class__.setValue
        if isinstance(widget, QLineEdit):
            return widget.__class__.text, widget.__class__.setText
        if isinstance(widget, SubAlgorithmWidget):
            return widget.__class__.get_values, widget.__class__.set_values
        if isinstance(widget, ListInput):
            return widget.__class__.get_value, widget.__class__.set_value
        if hasattr(widget, "get_value") and hasattr(widget, "set_value"):
            return widget.__class__.get_value, widget.__class__.set_value
        raise ValueError(f"Unsupported type: {type(widget)}")


class ListInput(QWidget):
    change_signal = Signal()

    def __init__(self, property_el: QtAlgorithmProperty, length):
        super().__init__()
        self.input_list = [property_el.from_algorithm_property(property_el) for _ in range(length)]
        layout = QVBoxLayout()
        for el in self.input_list:
            el.change_fun.connect(self.change_signal.emit)
            layout.addWidget(el.get_field())
        self.setLayout(layout)

    def get_value(self):
        return [x.get_value() for x in self.input_list]

    def set_value(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value for _ in range(len(self.input_list))]
        for f, val in zip(self.input_list, value):
            f.set_value(val)


def any_arguments(fun):
    def _any(*_):
        fun()

    return _any


class FormWidget(QWidget):
    value_changed = Signal()

    def __init__(self, fields: typing.List[AlgorithmProperty], start_values=None, dimension_num=1):
        super().__init__()
        if start_values is None:
            start_values = {}
        self.widgets_dict: typing.Dict[str, QtAlgorithmProperty] = dict()
        self.channels_chose: typing.List[typing.Union[ChannelComboBox, SubAlgorithmWidget]] = []
        layout = QFormLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        # layout.setVerticalSpacing(0)
        element_list = map(QtAlgorithmProperty.from_algorithm_property, fields)
        for el in element_list:
            if isinstance(el, QLabel):
                layout.addRow(el)
            elif isinstance(el.get_field(), SubAlgorithmWidget):
                label = QLabel(el.user_name)
                if el.help_text:
                    label.setToolTip(el.help_text)
                layout.addRow(label, el.get_field().choose)
                layout.addRow(el.get_field())
                self.widgets_dict[el.name] = el
                if el.name in start_values:
                    el.get_field().set_starting(start_values[el.name])
                el.change_fun.connect(any_arguments(self.value_changed.emit))
            else:
                self.widgets_dict[el.name] = el
                label = QLabel(el.user_name)
                if el.help_text:
                    label.setToolTip(el.help_text)
                layout.addRow(label, el.get_field())
                # noinspection PyUnresolvedReferences
                if issubclass(el.value_type, Channel):
                    # noinspection PyTypeChecker
                    self.channels_chose.append(el.get_field())
                if el.name in start_values:
                    try:
                        el.set_value(start_values[el.name])
                    except (KeyError, ValueError, TypeError):
                        pass
                el.change_fun.connect(any_arguments(self.value_changed.emit))
        self.setLayout(layout)
        self.value_changed.connect(self.update_size)

    def has_elements(self):
        return len(self.widgets_dict) > 0

    def update_size(self):
        self.setMinimumHeight(self.layout().minimumSize().height())

    def get_values(self):
        return {name: el.get_value() for name, el in self.widgets_dict.items()}

    def recursive_get_values(self):
        return {name: el.recursive_get_values() for name, el in self.widgets_dict.items()}

    def set_values(self, values: dict):
        for name, value in values.items():
            if name in self.widgets_dict:
                self.widgets_dict[name].set_value(value)

    def image_changed(self, image: Image):
        if not image:
            return
        for channel_widget in self.channels_chose:
            if isinstance(channel_widget, ChannelComboBox):
                channel_widget.change_channels_num(image.channels)
            else:
                channel_widget.change_channels_num(image)


class SubAlgorithmWidget(QWidget):
    values_changed = Signal()

    def __init__(self, algorithm_property: AlgorithmProperty):
        super().__init__()
        if not isinstance(algorithm_property.possible_values, dict):
            raise ValueError(
                "algorithm_property.possible_values should be dict." f"It is {type(algorithm_property.possible_values)}"
            )
        if not isinstance(algorithm_property.default_value, str):
            raise ValueError(
                "algorithm_property.default_value should be str." f"It is {type(algorithm_property.default_value)}"
            )
        self.starting_values = {}
        self.property = algorithm_property
        self.widgets_dict: typing.Dict[str, FormWidget] = {}
        # TODO protect for recursion
        widget = FormWidget(algorithm_property.possible_values[algorithm_property.default_value].get_fields())
        widget.layout().setContentsMargins(0, 0, 0, 0)
        widget.value_changed.connect(self.values_changed)

        self.widgets_dict[algorithm_property.default_value] = widget
        self.choose = QComboBox(self)
        self.choose.addItems(list(algorithm_property.possible_values.keys()))
        self.setContentsMargins(0, 0, 0, 0)

        self.choose.setCurrentText(algorithm_property.default_value)

        self.choose.currentTextChanged.connect(self.algorithm_choose)
        # self.setStyleSheet("border: 1px solid red")
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        if not widget.has_elements():
            widget.hide()
            self.hide()
        tmp_widget = QWidget(self)
        # tmp_widget.setMinimumHeight(5000)
        layout.addWidget(tmp_widget)
        self.tmp_widget = tmp_widget
        self.setLayout(layout)

    def set_starting(self, starting_values):
        self.starting_values = starting_values
        # self.set_values(starting_values)

    def set_values(self, val: dict):
        if not isinstance(val, dict):
            return
        self.choose.setCurrentText(val["name"])
        if val["name"] not in self.widgets_dict:
            self.algorithm_choose(val["name"])
        if val["name"] in self.widgets_dict:
            self.widgets_dict[val["name"]].set_values(val["values"])

    def recursive_get_values(self):
        return {name: el.recursive_get_values() for name, el in self.widgets_dict.items()}

    def get_values(self):
        name = self.choose.currentText()
        values = self.widgets_dict[name].get_values()
        return {"name": name, "values": values}

    def change_channels_num(self, image: Image):
        for i in range(self.layout().count()):
            el = self.layout().itemAt(i)
            if el.widget() and isinstance(el.widget(), FormWidget):
                el.widget().image_changed(image)

    def algorithm_choose(self, name):
        if name not in self.widgets_dict:
            if name not in self.property.possible_values:
                return
            start_dict = {} if name not in self.starting_values else self.starting_values[name]
            try:
                self.widgets_dict[name] = FormWidget(
                    self.property.possible_values[name].get_fields(), start_values=start_dict
                )
            except KeyError as e:
                raise e
            self.widgets_dict[name].layout().setContentsMargins(0, 0, 0, 0)
            self.layout().addWidget(self.widgets_dict[name])
            self.widgets_dict[name].value_changed.connect(self.values_changed)
        widget = self.widgets_dict[name]
        for i in range(self.layout().count()):
            lay_elem = self.layout().itemAt(i)
            if lay_elem.widget():
                lay_elem.widget().hide()
        if widget.has_elements():
            self.show()
            widget.show()
        else:
            self.hide()
        self.values_changed.emit()

    def showEvent(self, _event):
        # workaround for changing size
        self.tmp_widget.hide()

    def paintEvent(self, event: QPaintEvent):
        name = self.choose.currentText()
        if self.widgets_dict[name].has_elements() and event.rect().top() == 0 and event.rect().left() == 0:
            painter = QPainter(self)
            painter.drawRect(event.rect())


class BaseAlgorithmSettingsWidget(QScrollArea):
    values_changed = Signal()
    algorithm_thread: SegmentationThread

    def __init__(self, settings: BaseSettings, name, algorithm: typing.Type[SegmentationAlgorithm]):
        """
        For algorithm which works on one channel
        """
        super().__init__()
        self.settings = settings
        self.widget_list = []
        self.name = name
        self.algorithm = algorithm
        main_layout = QVBoxLayout()
        self.info_label = QLabel()
        self.info_label.setHidden(True)
        main_layout.addWidget(self.info_label)
        start_values = settings.get(f"algorithm_widget_state.{name}", {})
        self.form_widget = FormWidget(algorithm.get_fields(), start_values=start_values)
        self.form_widget.value_changed.connect(self.values_changed.emit)
        # self.form_widget.setMinimumHeight(1500)
        self.setWidget(self.form_widget)
        value_dict = self.settings.get(f"algorithms.{self.name}", {})
        self.set_values(value_dict)
        # self.settings.image_changed[Image].connect(self.image_changed)
        self.algorithm_thread = SegmentationThread(algorithm())
        self.algorithm_thread.info_signal.connect(self.show_info)
        self.algorithm_thread.exception_occurred.connect(self.exception_occurred)

    @staticmethod
    def exception_occurred(exc: Exception):
        if isinstance(exc, SegmentationLimitException):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("During segmentation process algorithm meet limitations:\n" + "\n".join(exc.args))
            mess.setWindowTitle("Segmentation limitations")
            mess.exec()
            return
        if isinstance(exc, RuntimeError) and exc.args[0].startswith(
            "Exception thrown in SimpleITK KittlerIllingworthThreshold"
        ):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("Fail to apply Kittler Illingworth to current data\n" + exc.args[0].split("\n")[1])
            mess.setWindowTitle("Segmentation limitations")
            mess.exec()
            return
        dial = ErrorDialog(exc, "Error during segmentation", f"{QApplication.instance().applicationName()}")
        dial.exec()

    def show_info(self, text):
        self.info_label.setText(text)
        self.info_label.setVisible(True)

    def image_changed(self, image: Image):
        self.form_widget.image_changed(image)
        self.algorithm_thread.algorithm.set_image(image)

    def set_mask(self, mask):
        self.algorithm_thread.algorithm.set_mask(mask)

    def set_values(self, values_dict):
        self.form_widget.set_values(values_dict)

    def get_values(self):
        return self.form_widget.get_values()

    def channel_num(self):
        return self.channels_chose.currentIndex()

    def execute(self, exclude_mask=None):
        values = self.get_values()
        self.settings.set(f"algorithms.{self.name}", deepcopy(values))
        self.algorithm_thread.set_parameters(**values)
        self.algorithm_thread.start()

    def hideEvent(self, a0: QHideEvent):
        self.algorithm_thread.clean()

    def resizeEvent(self, event: QResizeEvent):
        if self.height() < self.form_widget.height():
            self.setMinimumWidth(self.form_widget.width() + 20)
        else:
            self.setMinimumWidth(self.form_widget.width() + 10)
        super().resizeEvent(event)

    def recursive_get_values(self):
        return self.form_widget.recursive_get_values()


class AlgorithmSettingsWidget(BaseAlgorithmSettingsWidget):
    def execute(self, exclude_mask=None):
        self.algorithm_thread.algorithm.set_image(self.settings.image)
        super().execute(exclude_mask)


class InteractiveAlgorithmSettingsWidget(BaseAlgorithmSettingsWidget):
    algorithm_thread: SegmentationThread

    def __init__(self, settings, name, algorithm: typing.Type[SegmentationAlgorithm], selector: typing.List[QWidget]):
        super().__init__(settings, name, algorithm)
        self.selector = selector
        self.algorithm_thread.finished.connect(self.enable_selector)
        self.algorithm_thread.started.connect(self.disable_selector)
        # noinspection PyUnresolvedReferences
        if hasattr(settings, "mask_changed"):
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

    def get_segmentation_profile(self) -> ROIExtractionProfile:
        return ROIExtractionProfile("", self.algorithm.get_name(), self.get_values())


class AlgorithmChoose(QWidget):
    finished = Signal()
    started = Signal()
    result = Signal(SegmentationResult)
    value_changed = Signal()
    progress_signal = Signal(str, int)
    algorithm_changed = Signal(str)

    def __init__(
        self, settings: BaseSettings, algorithms: typing.Dict[str, typing.Type[SegmentationAlgorithm]], parent=None
    ):
        super().__init__(parent)
        self.settings = settings
        self.algorithms = algorithms
        settings.algorithm_changed.connect(self.updated_algorithm)
        self.stack_layout = QStackedLayout()
        self.algorithm_choose = QComboBox()
        self.algorithm_dict: typing.Dict[str, BaseAlgorithmSettingsWidget] = {}
        self.algorithm_choose.currentTextChanged.connect(self.change_algorithm)
        self.add_widgets_to_algorithm()

        self.settings.image_changed.connect(self.image_changed)
        # self.setMinimumWidth(370)

        self.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.algorithm_choose)
        layout.addLayout(self.stack_layout)
        self.setLayout(layout)

    def add_widgets_to_algorithm(self):
        self.algorithm_choose.blockSignals(True)
        self.algorithm_choose.clear()
        for name, val in self.algorithms.items():
            self.algorithm_choose.addItem(name)
            widget = InteractiveAlgorithmSettingsWidget(self.settings, name, val, [])
            self.algorithm_dict[name] = widget
            widget.algorithm_thread.execution_done.connect(self.result.emit)
            widget.algorithm_thread.finished.connect(self.finished.emit)
            widget.algorithm_thread.started.connect(self.started.emit)
            widget.algorithm_thread.progress_signal.connect(self.progress_signal.emit)
            widget.values_changed.connect(self.value_changed.emit)
            self.stack_layout.addWidget(widget)
        name = self.settings.get("current_algorithm", "")
        self.algorithm_choose.blockSignals(False)
        if name:
            self.algorithm_choose.setCurrentText(name)

    def reload(self, algorithms=None):
        if algorithms is not None:
            self.algorithms = algorithms
        for _ in range(self.stack_layout.count()):
            widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.takeAt(0).widget()
            widget.algorithm_thread.execution_done.disconnect()
            widget.algorithm_thread.finished.disconnect()
            widget.algorithm_thread.started.disconnect()
            widget.algorithm_thread.progress_signal.disconnect()
            widget.values_changed.disconnect()
        self.algorithm_dict = {}
        self.add_widgets_to_algorithm()

    def updated_algorithm(self):
        self.change_algorithm(
            self.settings.last_executed_algorithm,
            self.settings.get(f"algorithms.{self.settings.last_executed_algorithm}"),
        )

    def recursive_get_values(self):
        result = {key: widget.recursive_get_values() for key, widget in self.algorithm_dict.items()}

        self.settings.set("algorithm_widget_state", update(self.settings.get("algorithm_widget_state", dict), result))
        return result

    def change_algorithm(self, name, values: dict = None):
        self.settings.set("current_algorithm", name)
        widget = self.stack_layout.currentWidget()
        self.blockSignals(True)
        if name != widget.name:
            widget = self.algorithm_dict[name]
            self.stack_layout.setCurrentWidget(widget)
            widget.image_changed(self.settings.image)
            if hasattr(widget, "set_mask") and hasattr(self.settings, "mask"):
                widget.set_mask(self.settings.mask)
        elif values is None:
            self.blockSignals(False)
            return
        if values is not None:
            widget.set_values(values)
        self.algorithm_choose.setCurrentText(name)
        self.blockSignals(False)
        self.algorithm_changed.emit(name)

    def image_changed(self):
        current_widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.currentWidget()
        current_widget.image_changed(self.settings.image)
        if hasattr(self.settings, "mask") and hasattr(current_widget, "change_mask"):
            current_widget.change_mask()

    def mask_changed(self):
        current_widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.currentWidget()
        if hasattr(self.settings, "mask") and hasattr(current_widget, "change_mask"):
            current_widget.change_mask()

    def current_widget(self) -> InteractiveAlgorithmSettingsWidget:
        return self.stack_layout.currentWidget()

    def current_parameters(self) -> ROIExtractionProfile:
        widget = self.current_widget()
        return ROIExtractionProfile("", widget.name, widget.get_values())

    def get_info_text(self):
        return self.current_widget().algorithm_thread.get_info_text()


# AbstractAlgorithmSettingsWidget.register(AlgorithmSettingsWidget)
