from napari_plugin_engine import napari_hook_implementation

from . import segmentation
from .copy_labels import CopyLabelWidget
from .segmentation import gauss_background_estimate, laplacian_check, laplacian_estimate
from .verify_points import verify_segmentation

if "reload" in globals():
    import importlib

    importlib.reload(segmentation)

reload = False


def register():

    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    register_fun(segmentation.SMSegmentation, RegisterEnum.roi_analysis_segmentation_algorithm)


@napari_hook_implementation
def napari_experimental_provide_dock_widget():
    return CopyLabelWidget


@napari_hook_implementation
def napari_experimental_provide_function():
    return gauss_background_estimate  # , {"area": "bottom"}


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function2():
    return laplacian_check


@napari_hook_implementation(specname="napari_experimental_provide_function")
def napari_experimental_provide_function3():
    return laplacian_estimate


@napari_hook_implementation(specname="napari_experimental_provide_dock_widget")
def napari_experimental_provide_dock_widget2():
    return verify_segmentation, {"name": "Verify Segmentation"}
