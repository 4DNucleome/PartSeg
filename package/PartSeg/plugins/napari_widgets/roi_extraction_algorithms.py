import inspect
import itertools
import typing

import numpy as np
from magicgui.widgets import Widget, create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels, Layer
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.channel_class import Channel
from PartSegCore.mask.algorithm_description import mask_algorithm_dict
from PartSegCore.segmentation import ROIExtractionResult
from PartSegImage import Image

from ...common_gui.algorithms_description import (
    AlgorithmChooseBase,
    FormWidget,
    InteractiveAlgorithmSettingsWidget,
    QtAlgorithmProperty,
)
from ._settings import get_settings

if typing.TYPE_CHECKING:
    from qtpy.QtGui import QHideEvent, QShowEvent


class QtNapariAlgorithmProperty(QtAlgorithmProperty):
    def _get_field(self) -> typing.Union[QWidget, Widget]:
        if inspect.isclass(self.value_type) and issubclass(self.value_type, Channel):
            return create_widget(annotation=NapariImage, label="Image", options={})
        return super()._get_field()


class NapariFormWidget(FormWidget):
    @staticmethod
    def _element_list(fields) -> typing.Iterable[QtAlgorithmProperty]:
        mask = AlgorithmProperty("mask", "Mask", None, value_type=typing.Optional[Labels])
        return map(QtNapariAlgorithmProperty.from_algorithm_property, itertools.chain([mask], fields))

    def reset_choices(self, event=None):
        for widget in self.widgets_dict.values():
            # print("eeee", widget)
            if hasattr(widget.get_field(), "reset_choices"):
                widget.get_field().reset_choices(event)


class NapariInteractiveAlgorithmSettingsWidget(InteractiveAlgorithmSettingsWidget):
    @staticmethod
    def _form_widget(algorithm, start_values) -> FormWidget:
        return NapariFormWidget(algorithm.get_fields(), start_values=start_values)

    def reset_choices(self, event=None):
        self.form_widget.reset_choices(event)

    def get_order_mapping(self):
        layers = self.get_layers()
        layer_order_dict = {}
        for name, layer in layers.items():
            if isinstance(layer, NapariImage) and layer not in layer_order_dict:
                layer_order_dict[layer] = len(layer_order_dict)
        return layer_order_dict

    def get_values(self):
        layer_order_dict = self.get_order_mapping()
        return {
            k: layer_order_dict.get(v) if isinstance(v, NapariImage) else v
            for k, v in self.form_widget.get_values().items()
            if not isinstance(v, Labels) and k != "mask"
        }

    def get_layers(self):
        return {k: v for k, v in self.form_widget.get_values().items() if isinstance(v, Layer)}


class NapariAlgorithmChoose(AlgorithmChooseBase):
    @staticmethod
    def _algorithm_widget(settings, name, val) -> InteractiveAlgorithmSettingsWidget:
        return NapariInteractiveAlgorithmSettingsWidget(settings, name, val, [])

    def reset_choices(self, event=None):
        for widget in self.algorithm_dict.values():
            widget.reset_choices(event)


class ROIExtractionAlgorithms(QWidget):
    @staticmethod
    def get_method_dict():
        raise NotImplementedError

    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self._scale = 1, 1, 1
        self.channel_names = []
        self.mask_name = ""

        self.viewer = napari_viewer
        self.settings = get_settings()
        self.algorithm_chose = NapariAlgorithmChoose(self.settings, self.get_method_dict())
        self.calculate_btn = QPushButton("Run")
        self.calculate_btn.clicked.connect(self._run_calculation)

        layout = QVBoxLayout()
        layout.addWidget(self.calculate_btn)
        layout.addWidget(self.algorithm_chose)

        self.setLayout(layout)

        self.algorithm_chose.result.connect(self.set_result)
        self.algorithm_chose.algorithm_changed.connect(self.algorithm_changed)

    def algorithm_changed(self):
        self._scale = 1, 1, 1
        self.channel_names = []
        self.mask_name = ""

    def update_mask(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        mask = widget.get_layers().get("mask", None)
        if getattr(mask, "name", "") != self.mask_name:
            widget.set_mask(getattr(mask, "data", None))
            self.mask_name = getattr(mask, "name", "")

    def update_image(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        self.settings.last_executed_algorithm = widget.name
        layers = widget.get_order_mapping()
        axis_order = Image.axis_order.replace("C", "")
        channel_names = []
        for image_layer in layers:
            if image_layer.name not in channel_names:
                channel_names.append(image_layer.name)
        if self.channel_names == channel_names:
            return

        image_list = []
        for image_layer in layers:
            data_scale = image_layer.scale[-3:] / UNIT_SCALE[Units.nm.value]
            image_list.append(
                Image(
                    image_layer.data,
                    data_scale,
                    axes_order=axis_order[-image_layer.data.ndim :],
                    channel_names=[image_layer.name],
                )
            )
        res_image = image_list[0]
        for image in image_list[1:]:
            res_image = res_image.merge(image, "C")

        self._scale = np.array(res_image.spacing)
        self.channel_names = res_image.channel_names
        widget.image_changed(res_image)

    def _run_calculation(self):
        widget: NapariInteractiveAlgorithmSettingsWidget = self.algorithm_chose.current_widget()
        self.settings.last_executed_algorithm = widget.name
        self.update_image()
        self.update_mask()
        widget.execute()

    def showEvent(self, event: "QShowEvent") -> None:
        self.reset_choices(None)
        super().showEvent(event)

    def hideEvent(self, event: "QHideEvent") -> None:
        self.settings.dump()
        super().hideEvent(event)

    def reset_choices(self, event=None):
        self.algorithm_chose.reset_choices(event)

    def set_result(self, result: ROIExtractionResult):
        if "ROI" in self.viewer.layers:
            self.viewer.layers["ROI"].data = result.roi
        else:
            self.viewer.add_labels(
                result.roi, scale=self._scale[-result.roi.ndim :] * UNIT_SCALE[Units.nm.value], name="ROI"
            )


class ROIAnalysisExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return analysis_algorithm_dict


class ROIMaskExtraction(ROIExtractionAlgorithms):
    @staticmethod
    def get_method_dict():
        return mask_algorithm_dict
