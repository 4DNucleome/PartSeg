import warnings
from typing import Optional

from magicgui.widgets import CheckBox, Container, HBox, PushButton, Table, VBox, create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels as NapariLabels
from napari.qt import create_worker, progress
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtCore import QObject, Signal

from PartSegCore import UNIT_SCALE, Units
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, PerComponent
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile, MeasurementResult
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class _MainThreadMove(QObject):
    data_callback = Signal(object)

    def move(self, data):
        self.data_callback.emit(data)


class Measurement(Container):
    def __init__(self, napari_viewer: Viewer):
        super().__init__(layout="vertical")
        self.viewer = napari_viewer
        self.labels_choice = create_widget(annotation=NapariLabels, label="Labels")
        self.scale_units_select = create_widget(annotation=Units, label="Data units")  # EnumComboBox(Units)
        self.units_select = create_widget(annotation=Units, label="Units")
        self.image_choice = create_widget(annotation=Optional[NapariImage], label="Image")
        self.table = Table()
        self.calculate_btn = PushButton(text="Calculate")
        self.margins = (0, 0, 0, 0)
        self.progress = None
        self.prev_step = 0
        self._main_thread_move = _MainThreadMove()
        self._main_thread_move.data_callback.connect(self.step_changed)

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

        self.image_choice.native.currentIndexChanged.connect(self.refresh_measurements)
        self.labels_choice.changed.connect(self.refresh_measurements)
        self.calculate_btn.changed.connect(self.calculate)

    def calculate(self, event=None):
        to_calculate = []
        for chk in self.measurement_layout:
            # noinspection PyTypeChecker
            if chk.value:
                leaf: Leaf = MEASUREMENT_DICT[chk.text].get_starting_leaf()
                to_calculate.append(leaf.replace_(per_component=PerComponent.Yes, area=AreaType.ROI))
        if not to_calculate:
            warnings.warn("No measurement. Select at least one measurement")
            return

        profile = MeasurementProfile("", [MeasurementEntry(x.name, x) for x in to_calculate])

        data_layer = self.image_choice.value or self.labels_choice.value

        data_ndim = data_layer.data.ndim
        if data_ndim > 4:
            warnings.warn("Not Supported. Currently measurement engine does not support data over 4 dim (TZYX)")
            return
        data_scale = data_layer.scale[-3:] / UNIT_SCALE[self.scale_units_select.get_value().value]
        image = Image(data_layer.data, data_scale, axes_order="TZYX"[-data_ndim:])
        roi_info = ROIInfo(self.labels_choice.value.data).fit_to_image(image)

        self.progress = progress(total=len(profile.chosen_fields))
        self.prev_step = 0

        create_worker(
            profile.calculate,
            step_changed=self._main_thread_move.move,
            _start_thread=True,
            _connect={"returned": self.set_result, "finished": self.finished},
            image=image,
            channel_num=0,
            roi=roi_info,
            result_units=self.units_select.get_value(),
        )
        self.calculate_btn.enabled = False

    def step_changed(self, current_step: int):
        self.progress.update(current_step - self.prev_step)
        self.prev_step = current_step

    def set_result(self, result: MeasurementResult):
        self.table.value = result.to_dataframe()

    def finished(self):
        if self.progress:
            self.progress.close()
        self.calculate_btn.enabled = True

    def _clean_measurements(self):
        selected = set()
        for _ in range(len(self.measurement_layout)):
            # noinspection PyTypeChecker
            chk = self.measurement_layout.pop(0)
            if chk.value:
                selected.add(chk.text)
        return selected

    def refresh_measurements(self, event=None):
        has_channel = self.image_choice.value is not None
        selected = self._clean_measurements()
        for val in MEASUREMENT_DICT.values():
            area = val.get_starting_leaf().area
            pc = val.get_starting_leaf().per_component
            if (
                val.get_fields()
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
        self.refresh_measurements()

    def showEvent(self, event):
        self.reset_choices()
        self.refresh_measurements()


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return Measurement
