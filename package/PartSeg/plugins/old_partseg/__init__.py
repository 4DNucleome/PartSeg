from PartSeg.plugins.old_partseg.old_partseg import LoadPartSegOld


def register():
    from PartSeg import state_store  # noqa: PLC0415

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum  # noqa: PLC0415
        from PartSegCore.register import register as register_fun  # noqa: PLC0415

        register_fun(LoadPartSegOld, RegisterEnum.analysis_load)
