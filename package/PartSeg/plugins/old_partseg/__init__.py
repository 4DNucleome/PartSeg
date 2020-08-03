from .old_partseg import LoadPartSegOld


def register():
    from PartSegCore import state_store

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum, register

        register(LoadPartSegOld, RegisterEnum.analysis_load)
