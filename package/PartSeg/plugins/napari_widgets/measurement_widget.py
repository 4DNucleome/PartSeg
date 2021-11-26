from typing import Optional

from magicgui.widgets import create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels
from napari.utils.notifications import show_info
from qtpy.QtWidgets import QLabel, QTabWidget

from PartSeg._roi_analysis.advanced_window import MeasurementSettings
from PartSeg._roi_analysis.measurement_widget import NO_MEASUREMENT_STRING, FileNamesEnum, MeasurementWidgetBase
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSeg.plugins.napari_widgets.utils import NapariFormDialog, generate_image
from PartSegCore.analysis.measurement_calculation import MeasurementProfile, MeasurementResult
from PartSegCore.roi_info import ROIInfo


class NapariMeasurementSettings(MeasurementSettings):
    def form_dialog(self, arguments):
        return NapariFormDialog(arguments, settings=self.settings, parent=self)


class NapariMeasurementWidget(MeasurementWidgetBase):
    def __init__(self, settings, napari_viewer, segment=None):
        super().__init__(settings, segment)
        self.napari_viewer = napari_viewer
        self.channels_chose = create_widget(annotation=NapariImage, label="Image", options={})
        self.roi_chose = create_widget(annotation=Labels, label="ROI", options={})
        self.mask_chose = create_widget(annotation=Optional[Labels], label="ROI", options={})
        self.butt_layout3.insertWidget(0, QLabel("Channel:"))
        self.butt_layout3.insertWidget(1, self.channels_chose.native)
        self.butt_layout3.insertWidget(2, QLabel("ROI:"))
        self.butt_layout3.insertWidget(3, self.roi_chose.native)
        self.butt_layout3.insertWidget(4, QLabel("Mask:"))
        self.butt_layout3.insertWidget(5, self.mask_chose.native)
        self.file_names.setCurrentEnum(FileNamesEnum.No)
        self.file_names.setVisible(False)
        self.file_names_label.setVisible(False)

    def append_measurement_result(self):
        try:
            compute_class = self.settings.measurement_profiles[self.measurement_type.currentText()]
        except KeyError:
            show_info(f"Measurement profile '{self.measurement_type.currentText()}' not found")
            return
        if self.roi_chose.value is None:
            return
        if self.channels_chose.value is None:
            return
        for name in compute_class.get_channels_num():
            if name not in self.napari_viewer.layers:
                show_info("Cannot calculate this measurement because " f"image do not have layer {name}")
                return
        units = self.units_choose.currentEnum()
        image = generate_image(self.napari_viewer, self.channels_chose.value.name, *compute_class.get_channels_num())
        if self.mask_chose.value is not None:
            image.mask = self.mask_chose.value.data
        roi_info = ROIInfo(self.roi_chose.value.data).fit_to_image(image)
        dial = ExecuteFunctionDialog(
            compute_class.calculate,
            [image, self.channels_chose.value.name, roi_info, units],
            text="Measurement calculation",
            parent=self,
        )  # , exception_hook=exception_hook)
        dial.exec_()
        stat: MeasurementResult = dial.get_result()
        self.roi_chose.value.properties = stat.to_dataframe()
        if stat is None:
            return
        # stat.set_filename(self.settings.image_path)
        self.measurements_storage.add_measurements(stat)
        self.previous_profile = compute_class.name
        self.refresh_view()

    def check_if_measurement_can_be_calculated(self, name):
        if name in (NO_MEASUREMENT_STRING, ""):
            return NO_MEASUREMENT_STRING
        profile: MeasurementProfile = self.settings.measurement_profiles.get(name)
        if profile.is_any_mask_measurement() and self.mask_chose.value is None:
            show_info("To use this measurement set please use data with mask loaded")
            self.measurement_type.setCurrentIndex(0)
            return NO_MEASUREMENT_STRING
        if self.roi_chose.value is None:
            show_info("Before calculate measurement please select ROI Layer")
            self.measurement_type.setCurrentIndex(0)
            return NO_MEASUREMENT_STRING
        return name

    def reset_choices(self, event=None):
        self.channels_chose.reset_choices()
        self.roi_chose.reset_choices()
        self.mask_chose.reset_choices()

    def showEvent(self, event) -> None:
        self.reset_choices(None)
        super().showEvent(event)


class Measurement(QTabWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = get_settings()
        self.measurement_widget = NapariMeasurementWidget(self.settings, napari_viewer)
        self.measurement_settings = NapariMeasurementSettings(self.settings)
        self.addTab(self.measurement_widget, "Measurements")
        self.addTab(self.measurement_settings, "Measurements settings")

    def reset_choices(self, event=None):
        self.measurement_widget.reset_choices()
