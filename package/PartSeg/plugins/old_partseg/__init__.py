from .old_partseg import LoadPartSegOld


def register():
    from PartSegCore import state_store

    if state_store.custom_plugin_load:
        from PartSegCore.register import register, RegisterEnum

        register(LoadPartSegOld, RegisterEnum.analysis_load)
