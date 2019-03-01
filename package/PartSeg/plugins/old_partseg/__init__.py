from .old_partseg import LoadPartSegOld


def register():
    from PartSeg.utils import state_store
    if state_store.custom_plugin_load:
        from PartSeg.utils.register import register, RegisterEnum
        register(LoadPartSegOld, RegisterEnum.analysis_load)
