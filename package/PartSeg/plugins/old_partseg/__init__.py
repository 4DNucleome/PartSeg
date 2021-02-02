from .old_partseg import LoadPartSegOld


def register():
    from PartSegCore import state_store

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum
        from PartSegCore.register import register as register_fun

        register_fun(LoadPartSegOld, RegisterEnum.analysis_load)
