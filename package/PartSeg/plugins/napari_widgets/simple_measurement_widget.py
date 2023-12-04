"""Implementation of PartSeg measurement feature for napari viewer."""

import warnings
from typing import Optional

import numpy as np
from magicgui.widgets import CheckBox, Container, HBox, PushButton, Table, VBox, create_widget
from napari import Viewer
from napari._qt.qthreading import thread_worker
from napari.layers import Image as NapariImage
from napari.layers import Labels as NapariLabels
from napari.qt import create_worker

from PartSeg.plugins import register as register_plugins
from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import base_model_to_algorithm_property
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, PerComponent
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile, MeasurementResult
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class SimpleMeasurement(Container):
    """Widget that provide access to simple measurement feature from PartSeg ROI Mask."""

    def __init__(self, napari_viewer: Viewer):
        """:param napari_viewer: napari viewer instance."""
        super().__init__(layout="vertical")
        self.viewer = napari_viewer
        self.labels_choice = create_widget(annotation=NapariLabels, label="Labels")
        self.scale_units_select = create_widget(annotation=Units, label="Data units")  # EnumComboBox(Units)
        self.units_select = create_widget(annotation=Units, label="Units")
        self.image_choice = create_widget(annotation=Optional[NapariImage], label="Image", options={})
        self.table = Table()
        self.calculate_btn = PushButton(text="Calculate")
        self.margins = (0, 0, 0, 0)
        self.measurement_result: Optional[MeasurementResult] = None
        self.worker = None

        options_layout = HBox(
            widgets=(
                self.image_choice,
                self.labels_choice,
                self.scale_units_select,
                self.units_select,
                self.calculate_btn,
            )
        )
        self.measurement_layout = VBox()

        bottom_layout = HBox(widgets=(self.measurement_layout, self.table))

        self.insert(0, options_layout)
        self.insert(1, bottom_layout)

        self.image_choice.native.currentIndexChanged.connect(self._refresh_measurements)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self.labels_choice.changed.connect(self._refresh_measurements)
            self.calculate_btn.changed.connect(self._calculate)
        register_plugins()

    def _calculate(self, event=None):
        to_calculate = []
        for chk in self.measurement_layout:
            # noinspection PyTypeChecker
            if chk.value:
                leaf: Leaf = MEASUREMENT_DICT[chk.text].get_starting_leaf()
                to_calculate.append(leaf.replace_(per_component=PerComponent.Yes, area=AreaType.ROI))
        if not to_calculate:  # pragma: no cover
            warnings.warn("No measurement. Select at least one measurement", stacklevel=1)
            return

        profile = MeasurementProfile(
            name="", chosen_fields=[MeasurementEntry(name=x.name, calculation_tree=x) for x in to_calculate]
        )

        data_layer = self.image_choice.value or self.labels_choice.value

        data_ndim = data_layer.data.ndim
        if data_ndim > 4:  # pragma: no cover
            warnings.warn(
                "Not Supported. Currently measurement engine does not support data over 4 dim (TZYX)", stacklevel=1
            )
            return
        data_scale = data_layer.scale[-3:] / UNIT_SCALE[self.scale_units_select.get_value().value]
        image = Image(data_layer.data, data_scale, axes_order="TZYX"[-data_ndim:])
        worker = _prepare_data(profile, image, self.labels_choice.value.data)
        worker.returned.connect(self._calculate_next)
        worker.errored.connect(self._finished)
        worker.start()
        self.worker = worker
        self.calculate_btn.enabled = False

    def _calculate_next(self, data):
        profile, image, roi_info, segmentation_mask_map = data
        self.measurement_result = MeasurementResult(segmentation_mask_map)
        create_worker(
            profile.calculate_yield,
            _start_thread=True,
            _progress={"total": len(profile.chosen_fields)},
            _connect={"finished": self._finished, "yielded": self._set_result},
            image=image,
            channel_num=0,
            roi=roi_info,
            result_units=self.units_select.get_value(),
            segmentation_mask_map=segmentation_mask_map,
        )

    def _set_result(self, result):
        self.measurement_result[result[0]] = result[1]
        self.table.value = self.measurement_result.to_dataframe()

    def _finished(self):
        self.calculate_btn.enabled = True
        self.worker = None

    def _clean_measurements(self):
        selected = set()
        for _ in range(len(self.measurement_layout)):
            # noinspection PyTypeChecker
            chk = self.measurement_layout.pop(0)
            if chk.value:
                selected.add(chk.text)
        return selected

    def _refresh_measurements(self, event=None):
        has_channel = self.image_choice.value is not None
        selected = self._clean_measurements()
        for val in MEASUREMENT_DICT.values():
            area = val.get_starting_leaf().area
            pc = val.get_starting_leaf().per_component
            if (
                base_model_to_algorithm_property(val.__argument_class__)
                if val.__new_style__
                else val.get_fields()
                or (area is not None and area != AreaType.ROI)
                or (pc is not None and pc != PerComponent.Yes)
            ):
                continue
            text = val.get_name()
            chk = CheckBox(name=text, label=text)
            if val.need_channel() and not has_channel:
                chk.enabled = False
                chk.tooltip = "Need selected image"
            elif text in selected:
                chk.value = True
            self.measurement_layout.insert(-1, chk)

    def reset_choices(self, event=None):
        super().reset_choices(event)
        self._refresh_measurements()


@thread_worker(progress=True)
def _prepare_data(profile: MeasurementProfile, image: Image, labels: np.ndarray):
    roi_info = ROIInfo(labels).fit_to_image(image)
    yield 1
    segmentation_mask_map = profile.get_segmentation_mask_map(image, roi_info, time=0)
    yield 2
    return profile, image, roi_info, segmentation_mask_map
