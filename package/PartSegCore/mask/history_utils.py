from PartSegCore.mask.io_functions import MaskProjectTuple
from PartSegCore.mask_create import MaskProperty
from PartSegCore.project_info import HistoryElement


def create_history_element_from_segmentation_tuple(project_info: MaskProjectTuple, mask_property: MaskProperty):
    return HistoryElement.create(
        roi_info=project_info.roi_info,
        mask=project_info.mask,
        roi_extraction_parameters={
            "selected": project_info.selected_components,
            "parameters": project_info.roi_extraction_parameters,
        },
        mask_property=mask_property,
    )
