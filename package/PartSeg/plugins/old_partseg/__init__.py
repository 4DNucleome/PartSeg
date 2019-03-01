from .old_partseg import LoadPartSegOld


def register():
    from PartSeg.utils import report_utils
    if report_utils.custom_plugin_load:
        from PartSeg.utils.register import register, RegisterEnum
        register(LoadPartSegOld, RegisterEnum.analysis_load)
