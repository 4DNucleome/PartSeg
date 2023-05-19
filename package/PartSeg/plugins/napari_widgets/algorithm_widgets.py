import operator
from enum import Enum
from typing import Optional, Type

from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels, Layer
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QWidget

from PartSeg.plugins.napari_widgets.utils import NapariFormWidget
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import DoubleThresholdSelection, ThresholdSelection
from PartSegCore.utils import BaseModel


class CompareType(Enum):
    lower_threshold = 1
    upper_threshold = 2

    def __str__(self):
        return self.name.replace("_", " ").title()


class AlgModel(BaseModel):
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
        data = self.data.data
        mask = self.mask.data if self.mask is not None else None
        if mask is not None and mask.shape != data.shape:
            raise ValueError("Mask shape is different than data shape")
        op = operator.gt if self.operator == CompareType.lower_threshold else operator.lt
        data = self.threshold.algorithm().calculate_mask(
            data=data, mask=mask, arguments=self.threshold.values, operator=op
        )[0]
        return {"data": data, "meta": {"scale": self.data.scale}, "layer_type": "labels"}


class DoubleThresholdModel(ThresholdModel):
    threshold: DoubleThresholdSelection = DoubleThresholdSelection.get_default()


class NoiseFilteringModel(AlgModel):
    data: NapariImage
    noise_filtering: NoiseFilterSelection = NoiseFilterSelection.get_default()

    def run_calculation(self):
        data = self.data.data
        data = self.noise_filtering.algorithm().noise_filter(
            channel=data, spacing=self.data.scale[-3:], arguments=self.noise_filtering.values
        )
        return {
            "data": data,
            "meta": {"scale": self.data.scale, "contrast_limits": self.data.contrast_limits},
            "layer_type": "image",
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
        res = self.form.get_values().run_calculation()
        self.napari_viewer.add_layer(Layer.create(**res))


class Threshold(AlgorithmWidgetBase):
    __data_model__ = ThresholdModel


class DoubleThreshold(AlgorithmWidgetBase):
    __data_model__ = DoubleThresholdModel


class NoiseFiltering(AlgorithmWidgetBase):
    __data_model__ = NoiseFilteringModel
