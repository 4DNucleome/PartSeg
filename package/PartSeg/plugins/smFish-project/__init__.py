from . import segmentation

if "reload" in globals():
    import importlib

    importlib.reload(segmentation)

reload = False


def register():

    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    register_fun(segmentation.SMSegmentation, RegisterEnum.roi_analysis_segmentation_algorithm)
