from PartSeg.plugins.modeling_save.save_modeling_data import SaveModeling


def register():
    from PartSeg import state_store  # noqa: PLC0415

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum  # noqa: PLC0415
        from PartSegCore.register import register as register_fun  # noqa: PLC0415

        register_fun(SaveModeling, RegisterEnum.analysis_save)
