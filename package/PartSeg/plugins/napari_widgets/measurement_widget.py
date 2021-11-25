from magicgui.widgets import create_widget
from napari import Viewer
from napari.layers import Image as NapariImage
from napari.layers import Labels
from qtpy.QtWidgets import QLabel, QTabWidget

from PartSeg._roi_analysis.advanced_window import MeasurementSettings
from PartSeg._roi_analysis.measurement_widget import MeasurementWidgetBase
from PartSeg.plugins.napari_widgets._settings import get_settings
from PartSeg.plugins.napari_widgets.utils import NapariFormDialog


class NapariMeasurementSettings(MeasurementSettings):
    def form_dialog(self, arguments):
        return NapariFormDialog(arguments, settings=self.settings, parent=self)


class NapariMeasurementWidget(MeasurementWidgetBase):
    def __init__(self, settings, segment=None):
        super().__init__(settings, segment)
        self.channels_chose = create_widget(annotation=NapariImage, label="Image", options={})
        self.roi_chose = create_widget(annotation=Labels, label="ROI", options={})
        self.butt_layout3.insertWidget(0, QLabel("Channel:"))
        self.butt_layout3.insertWidget(1, self.channels_chose.native)
        self.butt_layout3.insertWidget(2, QLabel("ROI:"))
        self.butt_layout3.insertWidget(3, self.roi_chose.native)

    def append_measurement_result(self):
        pass
        # TODO implement

    def reset_choices(self, event=None):
        self.channels_chose.reset_choices()
        self.roi_chose.reset_choices()

    def showEvent(self, event) -> None:
        self.reset_choices(None)
        super().showEvent(event)


class MeasurementWidget(QTabWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = get_settings()
        self.measurement_widget = NapariMeasurementWidget(self.settings)
        self.measurement_settings = NapariMeasurementSettings(self.settings)
        self.addTab(self.measurement_widget, "Measurements")
        self.addTab(self.measurement_settings, "Measurements settings")

    def reset_choices(self, event=None):
        self.measurement_widget.reset_choices()
