import operator
from enum import Enum
from typing import Optional, Type

import numpy as np
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels, Layer
from napari.utils.notifications import show_info
from pydantic import ValidationError
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from PartSeg.plugins.napari_widgets.utils import NapariFormWidget
from PartSegCore.segmentation.border_smoothing import SmoothAlgorithmSelection
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import DoubleThresholdSelection, ThresholdSelection
from PartSegCore.segmentation.watershed import WatershedSelection
from PartSegCore.utils import BaseModel


class CompareType(Enum):
    lower_threshold = 1
    upper_threshold = 2

    def __str__(self):
        return self.name.replace("_", " ").title()


class AlgModel(BaseModel):
    layer_name: str = ""

    class Config(BaseModel.Config):
        arbitrary_types_allowed = True

    def run_calculation(self):
        raise NotImplementedError


class ThresholdModel(AlgModel):
    class Config(BaseModel.Config):
        arbitrary_types_allowed = True

    data: NapariImage
    mask: Optional[Labels] = None
    operator: CompareType = CompareType.lower_threshold
    threshold: ThresholdSelection = ThresholdSelection.get_default()

    def run_calculation(self):
        data = np.squeeze(self.data.data)
        mask = np.squeeze(self.mask.data) if self.mask is not None else None
        if mask is not None and mask.shape != data.shape:
            raise ValueError("Mask shape is different than data shape")
        op = operator.gt if self.operator == CompareType.lower_threshold else operator.lt
        res_data = self.threshold.algorithm().calculate_mask(
            data=data, mask=mask, arguments=self.threshold.values, operator=op
        )[0]
        return {
            "data": res_data.reshape(self.data.data.shape),
            "meta": {"scale": self.data.scale, "name": self.layer_name or None},
            "layer_type": "labels",
        }


class DoubleThresholdModel(ThresholdModel):
    threshold: DoubleThresholdSelection = DoubleThresholdSelection.get_default()


class NoiseFilterModel(AlgModel):
    data: NapariImage
    noise_filtering: NoiseFilterSelection = NoiseFilterSelection.get_default()

    def run_calculation(self):
        data = np.squeeze(self.data.data)
        res_data = self.noise_filtering.algorithm().noise_filter(
            channel=data, spacing=self.data.scale[-3:], arguments=self.noise_filtering.values
        )
        return {
            "data": res_data.reshape(self.data.data.shape),
            "meta": {
                "scale": self.data.scale,
                "contrast_limits": self.data.contrast_limits,
                "name": self.layer_name or None,
            },
            "layer_type": "image",
        }


class WatershedModel(AlgModel):
    data: NapariImage
    flow_area: Layer
    core_objects: Layer
    mask: Optional[Labels] = None
    flow_method: WatershedSelection = WatershedSelection.get_default()
    side_connection: bool = True
    operator: CompareType = CompareType.lower_threshold

    def run_calculation(self):
        data = self.data.data
        if self.flow_area is self.core_objects:
            flow_area = self.flow_area.data == 1
            core_objects = self.flow_area.data == 2
        else:
            flow_area = self.flow_area.data
            core_objects = self.core_objects.data
        components_num = np.amax(core_objects)
        op = operator.gt if self.operator == CompareType.lower_threshold else operator.lt

        data = self.flow_method.algorithm().calculate_mask(
            data=data,
            sprawl_area=flow_area,
            core_objects=core_objects,
            components_num=components_num,
            arguments=self.flow_method.values,
            spacing=self.data.scale[-3:],
            side_connection=self.side_connection,
            operator=op,
        )
        return {
            "data": data,
            "meta": {"scale": self.data.scale},
            "layer_type": "labels",
            "name": self.layer_name or None,
        }


class BorderSmoothingModel(AlgModel):
    data: Labels
    border_smoothing: SmoothAlgorithmSelection = SmoothAlgorithmSelection.get_default()
    only_side_connection: bool = True

    def run_calculation(self):
        data = self.data.data
        self.border_smoothing.algorithm().smooth(segmentation=np.squeeze(data), arguments=self.border_smoothing.values)
        return {
            "data": data.reshape(self.data.data.shape),
            "meta": {
                "scale": self.data.scale,
                "contrast_limits": self.data.contrast_limits,
                "name": self.layer_name or None,
            },
            "layer_type": "labels",
        }


class AlgorithmWidgetBase(QWidget):
    __data_model__: Type[BaseModel]

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)
        self.napari_viewer = napari_viewer

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_operation)

        self.form = NapariFormWidget(self.__data_model__)

        layout = QVBoxLayout()
        layout.addWidget(self.form)
        layout.addWidget(self.run_button)
        self.setLayout(layout)

    def reset_choices(self, event=None):
        self.form.reset_choices(event)

    def run_operation(self):
        try:
            res = self.form.get_values().run_calculation()
        except ValidationError:
            show_info("It looks like not all layers are selected")
            return
        layer = Layer.create(**res)
        if (
            layer.name in self.napari_viewer.layers
            and layer.__class__ is self.napari_viewer.layers[layer.name].__class__
            and np.array_equal(layer.scale, self.napari_viewer.layers[layer.name].scale)
        ):
            self.napari_viewer.layers[layer.name].data = res["data"]
        else:
            self.napari_viewer.add_layer(layer)


class Threshold(AlgorithmWidgetBase):
    __data_model__ = ThresholdModel


class DoubleThreshold(AlgorithmWidgetBase):
    __data_model__ = DoubleThresholdModel


class NoiseFilter(AlgorithmWidgetBase):
    __data_model__ = NoiseFilterModel


class BorderSmooth(AlgorithmWidgetBase):
    __data_model__ = BorderSmoothingModel
