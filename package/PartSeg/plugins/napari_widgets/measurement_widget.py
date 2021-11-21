from napari import Viewer
from qtpy.QtWidgets import QTabWidget

from PartSeg._roi_analysis.advanced_window import MeasurementSettings
from PartSeg._roi_analysis.measurement_widget import MeasurementWidget
from PartSeg.plugins.napari_widgets._settings import get_settings


class NapariMeasurementWidget(QTabWidget):
    def __init__(self, napari_viewer: Viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.settings = get_settings()
        self.measurement_widget = MeasurementWidget(self.settings)
        self.measurement_settings = MeasurementSettings(self.settings)
        self.addTab(self.measurement_widget, "Measurements")
        self.addTab(self.measurement_settings, "Measurements settings")
