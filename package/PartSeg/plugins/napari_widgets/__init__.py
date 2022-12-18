from napari_plugin_engine import napari_hook_implementation

from PartSeg._launcher.main_window import PartSegGUILauncher
from PartSeg.plugins import register as register_plugin
from PartSeg.plugins.napari_widgets.mask_create_widget import MaskCreate
from PartSeg.plugins.napari_widgets.roi_extraction_algorithms import ROIAnalysisExtraction, ROIMaskExtraction
from PartSeg.plugins.napari_widgets.search_label_widget import SearchLabel


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    from PartSeg.plugins.napari_widgets.simple_measurement_widget import SimpleMeasurement

    register_plugin()
    return SimpleMeasurement


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget1():
    register_plugin()
    return ROIAnalysisExtraction


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget2():
    register_plugin()
    return ROIMaskExtraction


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget3():
    return MaskCreate


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget4():
    from PartSeg.plugins.napari_widgets.measurement_widget import Measurement

    register_plugin()
    return Measurement


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget5():
    return SearchLabel


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget6():
    return PartSegGUILauncher
