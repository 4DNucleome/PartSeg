from napari_plugin_engine import napari_hook_implementation

from PartSeg.plugins.napari_widgets.roi_extraction_algorithms import ROIAnalysisExtraction, ROIMaskExtraction


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    from PartSeg.plugins.napari_widgets.measurement_widget import SimpleMeasurement

    return SimpleMeasurement


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget1():
    return ROIAnalysisExtraction


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget2():
    return ROIMaskExtraction
