from abc import ABCMeta, abstractmethod
from copy import deepcopy
from typing import Type, List

from PyQt5.QtGui import QHideEvent
from PyQt5.QtWidgets import QComboBox, QCheckBox, QWidget, QVBoxLayout, QLabel, QFormLayout, \
    QAbstractSpinBox, QScrollArea

from six import with_metaclass

from common_gui.dim_combobox import DimComboBox
from common_gui.universal_gui_part import CustomSpinBox, CustomDoubleSpinBox
from project_utils.segmentation.algorithm_base import SegmentationAlgorithm, AlgorithmProperty
from project_utils.error_dialog import ErrorDialog
from project_utils.image_operations import to_radius_type_dict, RadiusType
from project_utils.segmentation_thread import SegmentationThread
from project_utils.universal_const import UNIT_SCALE
from tiff_image import Image

from .settings import ImageSettings, BaseSettings


class QtAlgorithmProperty(AlgorithmProperty):
    qt_class_dict = {int: CustomSpinBox, float: CustomDoubleSpinBox, list: QComboBox, bool: QCheckBox,
                     RadiusType: DimComboBox}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def from_algorithm_property(cls, ob):
        """
        :type ob: AlgorithmProperty | str
        :param ob: AlgorithmProperty object
        :return: QtAlgorithmProperty | QLabel
        """
        if isinstance(ob, AlgorithmProperty):
            return cls(name=ob.name, user_name=ob.user_name, default_value=ob.default_value, options_range=ob.range,
                       single_steep=ob.single_step)
        elif isinstance(ob, str):
            return QLabel(ob)
        raise ValueError(f"unknown parameter type {type(ob)} of {ob}")

    def get_field(self):
        field = self.qt_class_dict[self.value_type]()
        if isinstance(field, DimComboBox):
            # noinspection PyTypeChecker
            field.setValue(self.default_value)
        elif isinstance(field, QComboBox):
            field.addItems(self.range)
            field.setCurrentIndex(self.range.index(self.default_value))
        elif isinstance(field, QCheckBox):
            field.setChecked(bool(self.default_value))
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


class BaseAlgorithmSettingsWidget(QScrollArea):
    algorithm_thread: SegmentationThread
    gauss_radius_name = "gauss_radius"
    use_gauss_name = "use_gauss"

    def __init__(self, settings: BaseSettings, name, algorithm: Type[SegmentationAlgorithm]):
        """
        For algorithm which works on one channel
        :type settings: ImageSettings
        :param element_list:
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
        widget_layout = QFormLayout()
        self.channels_chose = QComboBox()
        widget_layout.addRow("Channel", self.channels_chose)
        element_list = map(QtAlgorithmProperty.from_algorithm_property, algorithm.get_fields())
        for el in element_list:
            if isinstance(el, QLabel):
                widget_layout.addRow(el)
            else:
                self.widget_list.append((el.name, el.get_field()))
                widget_layout.addRow(el.user_name, self.widget_list[-1][-1])
        ww = QWidget()
        ww.setLayout(widget_layout)
        #self.setLayout(main_layout)
        self.setWidget(ww)
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
        ind = self.channels_chose.currentIndex()
        channels_num = image.channels
        self.algorithm_thread.algorithm.set_image(image)
        self.channels_chose.clear()
        self.channels_chose.addItems(map(str, range(channels_num)))
        if ind < 0 or ind > channels_num:
            ind = 0
        self.channels_chose.setCurrentIndex(ind)

    def set_values(self, values_dict):
        if "channel" in values_dict:
            self.channels_chose.setCurrentIndex(values_dict["channel"])
        for name, el in self.widget_list:
            if name not in values_dict:
                continue
            if isinstance(el, DimComboBox):
                el.setValue(values_dict[name])
            elif isinstance(el, QComboBox):
                el.setCurrentText(str(values_dict[name]))
            elif isinstance(el, QAbstractSpinBox):
                el.setValue(values_dict[name])
            elif isinstance(el, QCheckBox):
                el.setChecked(values_dict[name])
            else:
                raise ValueError("unsuported type {}".format(type(el)))

    def get_values(self):
        res = dict()
        for name, el in self.widget_list:
            if isinstance(el, DimComboBox):
                res[name] = el.value()
            if isinstance(el, QComboBox):
                res[name] = str(el.currentText())
            elif isinstance(el, QAbstractSpinBox):
                res[name] = el.value()
            elif isinstance(el, QCheckBox):
                res[name] = el.isChecked()
            else:
                raise ValueError("unsuported type {}".format(type(el)))
            # TODO mayby do it better. Maybe some special class for gauss choose
            if name == self.use_gauss_name:
                res[name] = to_radius_type_dict[res[name]]
        res["channel"] = self.channels_chose.currentIndex()
        return res

    def channel_num(self):
        return self.channels_chose.currentIndex()

    def execute(self, exclude_mask=None):
        values = self.get_values()
        self.settings.set(f"algorithms.{self.name}", deepcopy(values))
        scale = UNIT_SCALE[self.settings.get("units_index")]
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
        for _, el in self.widget_list:
            if isinstance(el, QAbstractSpinBox):
                el.valueChanged.connect(self.value_updated)
            elif isinstance(el, QCheckBox):
                el.stateChanged.connect(self.value_updated)
            elif isinstance(el, QComboBox):
                # noinspection PyUnresolvedReferences
                el.currentIndexChanged.connect(self.value_updated)
        # noinspection PyUnresolvedReferences
        self.channels_chose.currentIndexChanged.connect(self.value_updated)
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
        for el in  self.selector:
            el.setDisabled(True)

    def enable_selector(self):
        for el in  self.selector:
            el.setEnabled(True)


AbstractAlgorithmSettingsWidget.register(AlgorithmSettingsWidget)