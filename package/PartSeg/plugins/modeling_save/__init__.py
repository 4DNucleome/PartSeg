from .save_modeling_data import SaveModeling


def register():
    from PartSegCore import state_store

    if state_store.custom_plugin_load:
        from PartSegCore.register import RegisterEnum, register

        register(SaveModeling, RegisterEnum.analysis_save)
