import operator
from enum import Enum
from typing import Optional, Type

import numpy as np
import SimpleITK as sitk
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels, Layer
from napari.utils.notifications import show_info
from pydantic import ValidationError
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from PartSeg.plugins import register as register_plugins
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSeg.plugins.napari_widgets.utils import NapariFormWidget
from PartSegCore.segmentation.border_smoothing import SmoothAlgorithmSelection
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import DoubleThresholdSelection, ThresholdSelection
from PartSegCore.segmentation.watershed import WatershedSelection
from PartSegCore.utils import BaseModel
from PartSegImage.image import minimal_dtype


class CompareType(Enum):
    lower_threshold = 1
    upper_threshold = 2

    def __str__(self):
        return self.name.replace("_", " ").title()


class FlowType(Enum):
    bright_center = 1
    dark_center = 2

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
            "meta": {"scale": self.data.scale, "name": self.layer_name or "Threshold labels"},
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
                "name": self.layer_name or "Denoised image",
            },
            "layer_type": "image",
        }


def make_3d(data: np.ndarray):
    if data.ndim > 3:
        data = np.squeeze(data)
    if data.ndim == 2:
        return data.reshape((1, *data.shape))
    if data.ndim != 3:
        raise ValueError(f"Wrong data dimensionality with shape {data.shape}")
    return data


class WatershedModel(AlgModel):
    data: NapariImage
    flow_area: Labels
    core_objects: Labels
    mask: Optional[Labels] = None
    watershed: WatershedSelection = WatershedSelection.get_default()
    side_connection: bool = True
    operator: FlowType = FlowType.bright_center

    def run_calculation(self):
        data = make_3d(self.data.data)
        if self.flow_area is self.core_objects:
            flow_area = make_3d(self.flow_area.data == 1).astype(np.uint8)
            core_objects = sitk.GetArrayFromImage(
                sitk.ConnectedComponent(sitk.GetImageFromArray(make_3d(self.flow_area.data == 2).astype(np.uint8)))
            )
        else:
            flow_area = make_3d(self.flow_area.data)
            core_objects = make_3d(self.core_objects.data)
        components_num = np.amax(core_objects)
        core_objects = core_objects.astype(minimal_dtype(components_num))
        op = operator.gt if self.operator == FlowType.bright_center else operator.lt

        if op(1, 0):
            lower_bound = np.min(data[flow_area > 0])
            upper_bound = np.max(data[flow_area > 0])
        else:
            lower_bound = np.max(data[flow_area > 0])
            upper_bound = np.min(data[flow_area > 0])
        data = self.watershed.algorithm().sprawl(
            data=data,
            sprawl_area=flow_area,
            core_objects=core_objects,
            components_num=components_num,
            arguments=self.watershed.values,
            spacing=self.flow_area.scale[-3:],
            side_connection=self.side_connection,
            operator=op,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        return {
            "data": data.reshape(self.flow_area.data.shape),
            "meta": {
                "scale": self.flow_area.scale,
                "name": self.layer_name or "Watershed",
            },
            "layer_type": "labels",
        }


class BorderSmoothingModel(AlgModel):
    data: Labels
    border_smoothing: SmoothAlgorithmSelection = SmoothAlgorithmSelection.get_default()
    only_side_connection: bool = True

    def run_calculation(self):
        data = self.data.data
        res_data = self.border_smoothing.algorithm().smooth(
            segmentation=np.squeeze(data), arguments=self.border_smoothing.values
        )
        return {
            "data": res_data.reshape(self.data.data.shape),
            "meta": {
                "scale": self.data.scale,
                "name": self.layer_name or "Border smoothed",
            },
            "layer_type": "labels",
        }


class ConnectedComponentsModel(AlgModel):
    data: Labels
    side_connection: bool = True
    minimum_size: int = 20

    def run_calculation(self):
        data = np.squeeze(self.data.data)
        res_data = sitk.GetArrayFromImage(
            sitk.RelabelComponent(
                sitk.ConnectedComponent(sitk.GetImageFromArray(data), self.side_connection),
                self.minimum_size,
            )
        )
        return {
            "data": res_data.reshape(self.data.data.shape),
            "meta": {
                "scale": self.data.scale,
                "name": self.layer_name or "Connected components",
            },
            "layer_type": "labels",
        }


class SplitCoreObjectsModel(AlgModel):
    data: Labels

    def run_calculation(self):
        res_data = (self.data.data >= 2).astype(np.uint8)
        return {
            "data": res_data,
            "meta": {
                "scale": self.data.scale,
                "name": self.layer_name or "Core objects",
            },
            "layer_type": "labels",
        }


class AlgorithmWidgetBase(QWidget):
    __data_model__: Type[BaseModel]

    def __init__(self, napari_viewer: Viewer, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.napari_viewer = napari_viewer

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self.run_operation)

        self.form = NapariFormWidget(self.__data_model__)
        self.form.set_values(self.settings.get(f"widgets.{self.__class__.__name__}", {}))

        layout = QVBoxLayout()
        layout.addWidget(self.form)
        layout.addWidget(self.run_button)
        self.setLayout(layout)
        register_plugins()

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
        self.settings.set(f"widgets.{self.__class__.__name__}", dict(self.form.get_values()))
        self.settings.dump()


class Threshold(AlgorithmWidgetBase):
    __data_model__ = ThresholdModel


class DoubleThreshold(AlgorithmWidgetBase):
    __data_model__ = DoubleThresholdModel


class NoiseFilter(AlgorithmWidgetBase):
    __data_model__ = NoiseFilterModel


class BorderSmooth(AlgorithmWidgetBase):
    __data_model__ = BorderSmoothingModel


class Watershed(AlgorithmWidgetBase):
    __data_model__ = WatershedModel


class ConnectedComponents(AlgorithmWidgetBase):
    __data_model__ = ConnectedComponentsModel


class SplitCoreObjects(AlgorithmWidgetBase):
    __data_model__ = SplitCoreObjectsModel
