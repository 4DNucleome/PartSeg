from PartSeg.plugins.modeling_save.save_modeling_data import SaveModeling


def register():
    from PartSeg import state_store

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum
        from PartSegCore.register import register as register_fun

        register_fun(SaveModeling, RegisterEnum.analysis_save)
