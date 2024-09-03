import collections.abc
import inspect
import logging
import typing
from contextlib import suppress
from copy import deepcopy
from enum import Enum

import numpy as np
from magicgui.widgets import ComboBox, EmptyWidget, Widget, create_widget
from napari.layers.base import Layer
from pydantic import BaseModel
from qtpy.QtCore import QMargins, QObject, Signal
from qtpy.QtGui import QHideEvent, QPainter, QPaintEvent, QResizeEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)
from sentry_sdk.integrations.logging import ignore_logger
from superqt import QEnumComboBox

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_backend.segmentation_thread import SegmentationThread
from PartSeg.common_gui.error_report import ErrorDialog
from PartSeg.common_gui.universal_gui_part import ChannelComboBox, CustomDoubleSpinBox, CustomSpinBox, Hline
from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmProperty,
    AlgorithmSelection,
    ROIExtractionProfile,
    base_model_to_algorithm_property,
)
from PartSegCore.segmentation.algorithm_base import (
    ROIExtractionAlgorithm,
    ROIExtractionResult,
    SegmentationLimitException,
)
from PartSegImage import Channel, Image

try:
    from pydantic.fields import UndefinedType
except ImportError:  # pragma: no cover
    from pydantic_core import PydanticUndefinedType as UndefinedType

logger = logging.getLogger(__name__)
ignore_logger(__name__)


def recursive_update(d, u):
    if not isinstance(d, typing.MutableMapping):
        d = {}
    for k, v in u.items():
        d[k] = recursive_update(d.get(k, {}), v) if isinstance(v, collections.abc.Mapping) else v
    return d


def _pretty_print(data, indent=2) -> str:
    if isinstance(data, typing.Mapping):
        res = "\n"
        for k, v in data.items():
            res += f"{' ' * indent}{k}: {_pretty_print(v, indent+2)}\n"
        return res[:-1]
    return str(data)


class ProfileSelect(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._settings = None

    def _update_choices(self):
        if hasattr(self._settings, "roi_profiles"):
            self.clear()
            self.addItems(list(self._settings.roi_profiles.keys()))

    def set_settings(self, settings: BaseSettings):
        self._settings = settings
        if hasattr(self._settings, "roi_profiles"):
            self._settings.roi_profiles.setted.connect(self._update_choices)
            self._settings.roi_profiles.deleted.connect(self._update_choices)
        self._update_choices()

    def get_value(self):
        if self._settings is not None and hasattr(self._settings, "roi_profiles") and self.currentText():
            return self._settings.roi_profiles[self.currentText()]
        return None

    def set_value(self, value):
        if (
            self._settings is not None
            and hasattr(self._settings, "roi_profiles")
            and value.name in self._settings.roi_profiles
        ):
            self.setCurrentText(value.name)


class QtAlgorithmProperty(AlgorithmProperty):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._widget = self._get_field()
        self.change_fun = self.get_change_signal(self._widget)
        self._getter, self._setter = self.get_getter_and_setter_function(self._widget)
        with suppress(TypeError, ValueError):
            self._setter(self._widget, self.default_value)

    def get_value(self):
        return self._getter(self._widget)

    def is_multiline(self):
        return getattr(self._widget, "__multiline__", False)

    def recursive_get_values(self):
        if isinstance(self._widget, SubAlgorithmWidget):
            return self._widget.recursive_get_values()
        return self.get_value()

    def set_value(self, val):
        """set value of widget"""
        try:
            return self._setter(self._widget, val)
        except (TypeError, ValueError) as e:
            logger.error("Error %s setting value %s to %s", e, val, self.name)

    def get_field(self) -> QWidget:
        """
        Get representing widget
        :return:
        :rtype:
        """
        return self._widget

    @classmethod
    def from_algorithm_property(cls, ob: typing.Union[str, AlgorithmProperty]):
        """
        Create class instance base on :py:class:`.AlgorithmProperty` instance
        If ob is string equal to `hline` or that contains only
        `-` of length at least 5 then return :py:class:`.HLine`

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
                value_type=ob.value_type,
                possible_values=ob.possible_values,
                help_text=ob.help_text,
                per_dimension=ob.per_dimension,
                mgi_options=ob.mgi_options,
            )
        if isinstance(ob, str):
            if ob.lower() == "hline" or len(ob) > 5 and all(x == "-" for x in ob):
                return Hline()
            return QLabel(ob)
        raise ValueError(f"unknown parameter type {type(ob)} of {ob}")

    @staticmethod
    def _get_numeric_field(ap: AlgorithmProperty):
        if issubclass(ap.value_type, int):
            res = CustomSpinBox()
            if not isinstance(ap.default_value, int):
                raise ValueError(
                    f"Incompatible types. default_value should be type of int. Is {type(ap.default_value)}"
                )
        else:  # issubclass(ap.value_type, float):
            res = CustomDoubleSpinBox()
            if not isinstance(ap.default_value, (int, float)):
                raise ValueError(
                    f"Incompatible types. default_value should be type of float. Is {type(ap.default_value)}"
                )
        if ap.default_value < res.minimum():
            res.setMinimum(ap.default_value)
        if ap.range is not None:
            res.setRange(*ap.range)
        return res

    @classmethod
    def _get_field_from_value_type(cls, ap: AlgorithmProperty):
        if issubclass(ap.value_type, Channel):
            res = ChannelComboBox(parent=None)
            res.change_channels_num(10)
        elif issubclass(ap.value_type, AlgorithmDescribeBase):
            res = SubAlgorithmWidget(ap)
        elif issubclass(ap.value_type, bool):
            res = QCheckBox()
        elif issubclass(ap.value_type, (float, int)):
            res = cls._get_numeric_field(ap)
        elif issubclass(ap.value_type, Enum):
            # noinspection PyTypeChecker
            res = QEnumComboBox(enum_class=ap.value_type)
        elif issubclass(ap.value_type, str):
            res = QLineEdit()
        elif issubclass(ap.value_type, ROIExtractionProfile):
            res = ProfileSelect()
        elif issubclass(ap.value_type, list):
            res = QComboBox(parent=None)
            res.addItems(list(map(str, ap.possible_values)))
        elif issubclass(ap.value_type, BaseModel):
            res = FieldsList([cls.from_algorithm_property(x) for x in base_model_to_algorithm_property(ap.value_type)])
        else:
            res = cls._get_field_magicgui(ap)
        return res

    @classmethod
    def _get_field_magicgui(cls, ap: AlgorithmProperty) -> Widget:
        if isinstance(ap.default_value, UndefinedType) or ap.default_value is Ellipsis:
            res = create_widget(annotation=ap.value_type, options=ap.mgi_options)
        else:
            try:
                res = create_widget(value=ap.default_value, annotation=ap.value_type, options=ap.mgi_options)
            except ValueError as e:
                if "None is not a valid choice." in str(e):
                    res = create_widget(annotation=ap.value_type, options=ap.mgi_options)
                else:  # pragma: no cover
                    raise e

        if isinstance(res, EmptyWidget):  # pragma: no cover
            raise ValueError(f"Unknown type {ap.value_type}")
        return res

    def _get_field(self) -> typing.Union[QWidget, Widget]:
        """
        Get proper widget for given field type. Overwrite if would like to support new data types.
        """
        if self.per_dimension:
            self.per_dimension = False
            prop = self.from_algorithm_property(self)
            self.per_dimension = True
            res = ListInput(prop, 3)
        elif not inspect.isclass(self.value_type):
            res = self._get_field_magicgui(self)
        elif hasattr(self.value_type, "get_object"):
            res = self.value_type.get_object()
        else:
            res = self._get_field_from_value_type(self)
        tool_tip_text = self.help_text or ""
        tool_tip_text += f" default value: {_pretty_print(self.default_value)}"
        if isinstance(res, QWidget):
            res.setToolTip(tool_tip_text)
        if isinstance(res, Widget):
            res.tooltip = tool_tip_text  # pylint: disable=attribute-defined-outside-init # false positive
        return res

    @staticmethod
    def get_change_signal(widget: typing.Union[QWidget, Widget]):  # noqa: PLR0911
        if isinstance(widget, Widget):
            return widget.changed
        if isinstance(widget, QComboBox):
            return widget.currentIndexChanged
        if isinstance(widget, QCheckBox):
            return widget.stateChanged
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.valueChanged
        if isinstance(widget, QLineEdit):
            return widget.textChanged
        if isinstance(widget, SubAlgorithmWidget):
            return widget.values_changed
        if isinstance(widget, ListInput):
            return widget.change_signal
        if isinstance(widget, FieldsList):
            return widget.changed
        if hasattr(widget, "values_changed"):
            return widget.values_changed
        raise ValueError(f"Unsupported type: {type(widget)}")

    @staticmethod
    def get_getter_and_setter_function(  # noqa: PLR0911
        widget: typing.Union[QWidget, Widget],
    ) -> typing.Tuple[
        typing.Callable[
            [typing.Union[QWidget, Widget]],
            typing.Any,
        ],
        typing.Callable[[typing.Union[QWidget, Widget], typing.Any], None],
    ]:
        """
        For each widget type return proper functions. This functions need instance as first argument

        :return: (getter, setter)
        """
        if isinstance(widget, Widget):
            return _value_get, _value_set
        if isinstance(widget, (ProfileSelect, ChannelComboBox)):
            return widget.__class__.get_value, widget.__class__.set_value
        if isinstance(widget, QEnumComboBox):
            return widget.__class__.currentEnum, widget.__class__.setCurrentEnum
        if isinstance(widget, QComboBox):
            return widget.__class__.currentText, widget.__class__.setCurrentText
        if isinstance(widget, QCheckBox):
            return widget.__class__.isChecked, widget.__class__.setChecked
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.__class__.value, widget.__class__.setValue
        if isinstance(widget, QLineEdit):
            return widget.__class__.text, widget.__class__.setText
        if isinstance(widget, SubAlgorithmWidget):
            return widget.__class__.get_values, widget.__class__.set_values
        if isinstance(widget, ListInput):
            return widget.__class__.get_value, widget.__class__.set_value
        if hasattr(widget.__class__, "get_value") and hasattr(widget.__class__, "set_value"):
            return widget.__class__.get_value, widget.__class__.set_value
        raise ValueError(f"Unsupported type: {type(widget)}")


class FieldsList(QObject):
    changed = Signal()

    def __init__(self, field_list: typing.List[QtAlgorithmProperty]):
        super().__init__()
        self.field_list = field_list
        for el in field_list:
            el.change_fun.connect(self._changed_wrap)

    def get_value(self):
        return {el.name: el.get_value() for el in self.field_list}

    def _changed_wrap(self, val=None):
        self.changed.emit()

    def set_value(self, val):
        if isinstance(val, dict):
            self._set_value_dkt(val)
        else:
            self._set_value_base_model(val)

    def _set_value_base_model(self, val):
        for el in self.field_list:
            if hasattr(val, el.name):
                el.set_value(getattr(val, el.name))

    def _set_value_dkt(self, val: dict):
        for el in self.field_list:
            if el.name in val:
                el.set_value(val[el.name])


class ListInput(QWidget):
    change_signal = Signal()

    def __init__(self, property_el: QtAlgorithmProperty, length):
        super().__init__()
        self.input_list = [property_el.from_algorithm_property(property_el) for _ in range(length)]
        layout = QVBoxLayout()
        for el in self.input_list:
            el.change_fun.connect(_any_arguments(self.change_signal.emit))
            layout.addWidget(el.get_field())
        self.setLayout(layout)

    def get_value(self):
        return [x.get_value() for x in self.input_list]

    def set_value(self, value):
        if not isinstance(value, (list, tuple)):
            value = [value for _ in range(len(self.input_list))]
        for f, val in zip(self.input_list, value):
            f.set_value(val)


def _any_arguments(fun):
    def _any():
        fun()

    return _any


FieldAllowedTypes = typing.Union[
    typing.List[AlgorithmProperty], typing.Type[BaseModel], typing.Type[AlgorithmDescribeBase]
]


class FormWidget(QWidget):
    value_changed = Signal()

    def __init__(
        self,
        fields: FieldAllowedTypes,
        start_values=None,
        dimension_num=1,
        settings: typing.Optional[BaseSettings] = None,
        parent=None,
    ):
        super().__init__(parent=parent)
        if start_values is None:
            start_values = {}
        self.widgets_dict: typing.Dict[str, QtAlgorithmProperty] = {}
        self.channels_chose: typing.List[typing.Union[ChannelComboBox, SubAlgorithmWidget]] = []
        layout = QFormLayout()
        layout.setContentsMargins(10, 0, 10, 0)
        self._model_class = None
        element_list = self._element_list(fields)
        for el in element_list:
            if isinstance(el, (QLabel, Hline)):
                layout.addRow(el)
                continue
            self._add_to_layout(layout, el, start_values, settings)
            if hasattr(el.get_field(), "change_channels_num"):
                self.channels_chose.append(el.get_field())
        self.setLayout(layout)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)

    def _element_list(self, fields: FieldAllowedTypes):
        if inspect.isclass(fields) and issubclass(fields, AlgorithmDescribeBase):
            if fields.__new_style__:
                self._model_class = fields.__argument_class__
                fields = base_model_to_algorithm_property(fields.__argument_class__)
            else:
                fields = fields.get_fields()
        elif not isinstance(fields, typing.Iterable):
            self._model_class = fields
            fields = base_model_to_algorithm_property(fields)
        return self._element_list_map(fields)

    @staticmethod
    def _element_list_map(fields):
        return map(QtAlgorithmProperty.from_algorithm_property, fields)

    def _add_to_layout(
        self, layout, ap: QtAlgorithmProperty, start_values: typing.MutableMapping, settings, add_to_widget_dict=True
    ):
        label = QLabel(ap.user_name)
        if ap.help_text:
            label.setToolTip(ap.help_text)
        if add_to_widget_dict:
            self.widgets_dict[ap.name] = ap
        ap.change_fun.connect(_any_arguments(self.value_changed.emit))
        if isinstance(ap.get_field(), SubAlgorithmWidget):
            w = typing.cast(SubAlgorithmWidget, ap.get_field())
            layout.addRow(label, w.choose)
            layout.addRow(ap.get_field())
            if ap.name in start_values:
                w.set_starting(start_values[ap.name])
            ap.change_fun.connect(_any_arguments(self.value_changed.emit))
            return
        if isinstance(ap.get_field(), FieldsList):
            layout.addRow(label)
            for el in typing.cast(FieldsList, ap.get_field()).field_list:
                self._add_to_layout(layout, el, start_values.get(ap.name, {}), settings, add_to_widget_dict=False)
            return
        if isinstance(ap.get_field(), Widget):
            widget = typing.cast(Widget, ap.get_field()).native
        else:
            widget = ap.get_field()
        if ap.is_multiline():
            layout.addRow(label)
            layout.addRow(widget)
        else:
            layout.addRow(label, widget)
        # noinspection PyUnresolvedReferences
        if inspect.isclass(ap.value_type) and issubclass(ap.value_type, ROIExtractionProfile):
            # noinspection PyTypeChecker
            ap.get_field().set_settings(settings)
        if ap.name in start_values:
            with suppress(KeyError, ValueError, TypeError):
                ap.set_value(start_values[ap.name])

    def has_elements(self):
        return len(self.widgets_dict) > 0

    def get_values(self):
        res = {name: el.get_value() for name, el in self.widgets_dict.items()}
        return self._model_class(**res) if self._model_class is not None else res

    def recursive_get_values(self):
        return {name: el.recursive_get_values() for name, el in self.widgets_dict.items()}

    def set_values(self, values: typing.Union[dict, BaseModel]):
        if isinstance(values, BaseModel):
            values = dict(values)
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
        if not isinstance(algorithm_property.possible_values, typing.MutableMapping):
            raise ValueError(
                f"algorithm_property.possible_values should be dict. It is {type(algorithm_property.possible_values)}"
            )
        if not isinstance(algorithm_property.default_value, str):
            raise ValueError(
                f"algorithm_property.default_value should be str. It is {type(algorithm_property.default_value)}"
            )
        self.starting_values = {}
        self.property = algorithm_property
        self.widgets_dict: typing.Dict[str, FormWidget] = {}
        # TODO protect for recursion
        widget = self._get_form_widget(algorithm_property)
        widget.value_changed.connect(self.values_changed)

        self.widgets_dict[algorithm_property.default_value] = widget
        self.choose = QComboBox(self)
        self.choose.addItems(list(algorithm_property.possible_values.keys()))
        self.setContentsMargins(0, 0, 0, 0)

        self.choose.setCurrentText(algorithm_property.default_value)

        self.choose.currentTextChanged.connect(self.algorithm_choose)
        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(widget)
        if not widget.has_elements():
            widget.hide()
            self.hide()
        tmp_widget = QWidget(self)
        layout.addWidget(tmp_widget)
        self.tmp_widget = tmp_widget
        self.setLayout(layout)

    @staticmethod
    def _get_form_widget(algorithm_property, start_values=None):
        if isinstance(algorithm_property, AlgorithmProperty):
            calc_class = algorithm_property.possible_values[algorithm_property.default_value]
        else:
            calc_class = algorithm_property
        widget = FormWidget(calc_class, start_values=start_values)
        widget.layout().setContentsMargins(0, 0, 0, 0)
        return widget

    def set_starting(self, starting_values):
        self.starting_values = starting_values

    def set_values(self, val: typing.Mapping):
        if isinstance(val, BaseModel):
            val = dict(val)
        if not isinstance(val, typing.Mapping):
            return
        self.choose.setCurrentText(val["name"])
        if val["name"] not in self.widgets_dict:
            self.algorithm_choose(val["name"])
        if val["name"] in self.widgets_dict:
            self.widgets_dict[val["name"]].set_values(val["values"])

    def recursive_get_values(self):
        return {name: el.recursive_get_values() for name, el in self.widgets_dict.items()}

    def get_values(self) -> typing.Dict[str, typing.Any]:
        name = self.choose.currentText()
        values = self.widgets_dict[name].get_values()
        return {"name": name, "values": values}

    def change_channels_num(self, image: Image):
        for i in range(self.layout().count()):
            el = self.layout().itemAt(i)
            if el.widget() and isinstance(el.widget(), FormWidget):
                typing.cast(FormWidget, el.widget()).image_changed(image)

    def algorithm_choose(self, name):
        if name not in self.widgets_dict:
            if name not in self.property.possible_values:
                return
            start_dict = self.starting_values.get(name, {})
            self.widgets_dict[name] = self._get_form_widget(self.property.possible_values[name], start_dict)

            self.layout().addWidget(self.widgets_dict[name])
            self.widgets_dict[name].value_changed.connect(self.values_changed)
        widget = self.widgets_dict[name]
        for i in range(self.layout().count()):
            lay_elem = self.layout().itemAt(i)
            if lay_elem.widget():
                lay_elem.widget().setVisible(False)
        if widget.has_elements() and self.parent() is not None:
            self.setVisible(True)
            widget.setVisible(True)
        else:
            self.setVisible(False)
        self.values_changed.emit()

    def showEvent(self, _event):
        # workaround for changing size
        self.tmp_widget.hide()

    def paintEvent(self, event: QPaintEvent):
        name = self.choose.currentText()
        if (
            name in self.widgets_dict
            and self.widgets_dict[name].has_elements()
            and event.rect().top() == 0
            and event.rect().left() == 0
        ):
            painter = QPainter(self)
            painter.drawRect(self.rect() - QMargins(1, -1, 1, 1))


class BaseAlgorithmSettingsWidget(QScrollArea):
    values_changed = Signal()
    algorithm_thread: SegmentationThread

    def __init__(self, settings: BaseSettings, algorithm: typing.Type[ROIExtractionAlgorithm], parent=None):
        """
        For algorithm which works on one channel
        """
        super().__init__(parent=parent)
        self.settings = settings
        self.widget_list = []
        self.algorithm = algorithm
        main_layout = QVBoxLayout()
        self.info_label = QLabel()
        self.info_label.setHidden(True)
        # FIXME verify inflo_label usage
        main_layout.addWidget(self.info_label)
        start_values = settings.get_algorithm(f"algorithm_widget_state.{self.name}", {})
        self.form_widget = self._form_widget(algorithm, start_values=start_values)
        self.form_widget.value_changed.connect(self.values_changed.emit)
        self._widget = QWidget(self)
        self._widget.setLayout(QVBoxLayout())
        self._widget.layout().setContentsMargins(0, 0, 0, 0)
        self._widget.layout().addWidget(self.form_widget)
        self.setWidget(self._widget)
        value_dict = self.settings.get_algorithm(f"algorithms.{self.name}", {})
        self.set_values(value_dict)
        self.algorithm_thread = SegmentationThread(algorithm())
        self.algorithm_thread.info_signal.connect(self.show_info)
        self.algorithm_thread.exception_occurred.connect(self.exception_occurred)
        self.setWidgetResizable(True)

    @property
    def name(self):
        return self.algorithm.get_name()

    def _form_widget(self, algorithm, start_values) -> FormWidget:
        return FormWidget(
            algorithm,
            start_values=start_values,
            parent=self,
        )

    @staticmethod
    def exception_occurred(exc: Exception):
        if isinstance(exc, SegmentationLimitException):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("During segmentation process algorithm meet limitations:\n" + "\n".join(exc.args))
            mess.setWindowTitle("Segmentation limitations")
            mess.exec_()
            return
        if isinstance(exc, RuntimeError) and exc.args[0].startswith(
            "Exception thrown in SimpleITK KittlerIllingworthThreshold"
        ):
            mess = QMessageBox()
            mess.setIcon(QMessageBox.Critical)
            mess.setText("Fail to apply Kittler Illingworth to current data\n" + exc.args[0].split("\n")[1])
            mess.setWindowTitle("Segmentation limitations")
            mess.exec_()
            return
        dial = ErrorDialog(exc, "Error during segmentation", f"{QApplication.instance().applicationName()}")
        dial.exec_()

    def show_info(self, text):
        self.info_label.setText(text)
        self.info_label.setVisible(bool(text))

    def image_changed(self, image: Image):
        self.form_widget.image_changed(image)
        self.algorithm_thread.set_image(image)

    def set_mask(self, mask):
        self.algorithm_thread.set_mask(mask)

    def mask(self) -> typing.Optional[np.ndarray]:
        return self.algorithm_thread.algorithm.mask

    def set_values(self, values_dict):
        self.form_widget.set_values(values_dict)

    def get_values(self):
        return self.form_widget.get_values()

    def execute(self, exclude_mask=None):
        values = self.get_values()
        self.settings.set_algorithm(f"algorithms.{self.name}", deepcopy(values))
        if isinstance(values, dict):
            self.algorithm_thread.set_parameters(**values)
        else:
            self.algorithm_thread.set_parameters(values)
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


class InteractiveAlgorithmSettingsWidget(BaseAlgorithmSettingsWidget):
    algorithm_thread: SegmentationThread

    def __init__(
        self, settings, algorithm: typing.Type[ROIExtractionAlgorithm], selector: typing.List[QWidget], parent=None
    ):
        super().__init__(settings, algorithm, parent=parent)
        self.selector = selector[:]
        self.algorithm_thread.finished.connect(self.enable_selector)
        self.algorithm_thread.started.connect(self.disable_selector)
        # noinspection PyUnresolvedReferences
        if hasattr(settings, "mask_changed"):
            settings.mask_changed.connect(self.change_mask)

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
        return ROIExtractionProfile(name="", algorithm=self.algorithm.get_name(), values=self.get_values())


class AlgorithmChooseBase(QWidget):
    finished = Signal()
    started = Signal()
    result = Signal(ROIExtractionResult)
    value_changed = Signal()
    progress_signal = Signal(str, int)
    algorithm_changed = Signal(str)

    algorithm_dict: typing.Dict[str, InteractiveAlgorithmSettingsWidget]

    def __init__(self, settings: BaseSettings, algorithms: typing.Type[AlgorithmSelection], parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.algorithms = algorithms
        settings.algorithm_changed.connect(self.updated_algorithm)
        self.stack_layout = QStackedLayout()
        self.algorithm_choose = QComboBox()
        self.algorithm_dict: typing.Dict[str, BaseAlgorithmSettingsWidget] = {}
        self.algorithm_choose.currentTextChanged.connect(self.change_algorithm)
        self.add_widgets_to_algorithm()

        self.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.algorithm_choose)
        layout.addLayout(self.stack_layout)
        self.setLayout(layout)

    def _algorithm_widget(self, settings, val) -> InteractiveAlgorithmSettingsWidget:
        return InteractiveAlgorithmSettingsWidget(settings, val, [], parent=self)

    def add_widgets_to_algorithm(self):
        self.algorithm_choose.blockSignals(True)
        self.algorithm_choose.clear()
        for name, val in self.algorithms.__register__.items():
            self.algorithm_choose.addItem(name)
            widget = self._algorithm_widget(self.settings, val)
            self.algorithm_dict[name] = widget
            widget.algorithm_thread.execution_done.connect(self.result.emit)
            widget.algorithm_thread.finished.connect(self.finished.emit)
            widget.algorithm_thread.started.connect(self.started.emit)
            widget.algorithm_thread.progress_signal.connect(self.progress_signal.emit)
            widget.values_changed.connect(self.value_changed.emit)
            self.stack_layout.addWidget(widget)
        name = self.settings.get_algorithm("current_algorithm", "")
        self.algorithm_choose.blockSignals(False)
        if name:
            self.algorithm_choose.setCurrentText(name)

    def reload(self, algorithms=None):
        if algorithms is not None:
            self.algorithms = algorithms
        for _ in range(self.stack_layout.count()):
            widget = typing.cast(InteractiveAlgorithmSettingsWidget, self.stack_layout.takeAt(0).widget())
            widget.algorithm_thread.execution_done.disconnect()
            widget.algorithm_thread.finished.disconnect()
            widget.algorithm_thread.started.disconnect()
            widget.algorithm_thread.progress_signal.disconnect()
            widget.values_changed.disconnect()
            widget.deleteLater()
        self.algorithm_dict = {}
        self.add_widgets_to_algorithm()

    def updated_algorithm(self):
        self.change_algorithm(
            self.settings.last_executed_algorithm,
            self.settings.get_algorithm(f"algorithms.{self.settings.last_executed_algorithm}"),
        )

    def recursive_get_values(self):
        result = {key: widget.recursive_get_values() for key, widget in self.algorithm_dict.items()}

        self.settings.set_algorithm(
            "algorithm_widget_state",
            recursive_update(self.settings.get_algorithm("algorithm_widget_state", {}), result),
        )
        return result

    def change_algorithm(self, name, values: typing.Optional[dict] = None):
        self.settings.set_algorithm("current_algorithm", name)
        widget = typing.cast(InteractiveAlgorithmSettingsWidget, self.stack_layout.currentWidget())
        blocked = self.blockSignals(True)
        if name != widget.name:
            widget = self.algorithm_dict[name]
            self.stack_layout.setCurrentWidget(widget)
            widget.image_changed(self.settings.image)
            if hasattr(widget, "set_mask") and hasattr(self.settings, "mask"):
                widget.set_mask(self.settings.mask)
        elif values is None:
            self.blockSignals(blocked)
            return
        if values is not None:
            widget.set_values(values)
        self.algorithm_choose.setCurrentText(name)
        self.blockSignals(blocked)
        self.algorithm_changed.emit(name)

    def current_widget(self) -> InteractiveAlgorithmSettingsWidget:
        return typing.cast(InteractiveAlgorithmSettingsWidget, self.stack_layout.currentWidget())

    def current_parameters(self) -> ROIExtractionProfile:
        widget = self.current_widget()
        return ROIExtractionProfile(name="", algorithm=widget.name, values=widget.get_values())

    def get_info_text(self):
        return self.current_widget().algorithm_thread.get_info_text()


class AlgorithmChoose(AlgorithmChooseBase):
    def __init__(self, settings: BaseSettings, algorithms: typing.Type[AlgorithmSelection], parent=None):
        super().__init__(settings, algorithms, parent)
        self.settings.image_changed.connect(self.image_changed)

    def image_changed(self):
        current_widget = typing.cast(InteractiveAlgorithmSettingsWidget, self.stack_layout.currentWidget())
        prev_block = self.blockSignals(True)
        current_widget.image_changed(self.settings.image)
        if hasattr(self.settings, "mask") and hasattr(current_widget, "change_mask"):
            current_widget.change_mask()
        self.blockSignals(prev_block)
        self.value_changed.emit()


def _value_get(self: Widget):
    return self.value


def _value_set(self: Widget, value: typing.Any):
    if isinstance(self, ComboBox) and issubclass(self.annotation, Layer) and isinstance(value, str):
        for el in self.choices:
            if el.name == value:
                self.value = el
                return
    if isinstance(value, Channel):
        self.value = value.value
    self.value = value
