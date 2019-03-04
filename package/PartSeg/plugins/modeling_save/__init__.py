from .save_modeling_data import SaveModeling

def register():
    from PartSeg.utils import state_store
    if state_store.custom_plugin_load:
        from PartSeg.utils.register import register, RegisterEnum
        register(SaveModeling, RegisterEnum.analysis_save)