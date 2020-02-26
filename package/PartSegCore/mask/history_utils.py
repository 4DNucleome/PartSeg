from PartSegCore.io_utils import HistoryElement
from PartSegCore.mask.io_functions import SegmentationTuple
from PartSegCore.mask_create import MaskProperty


def create_history_element_from_segmentation_tuple(project_info: SegmentationTuple, mask_property: MaskProperty):
    return HistoryElement.create(
        segmentation=project_info.segmentation,
        full_segmentation=project_info.segmentation,
        mask=project_info.mask,
        segmentation_parameters={
            "selected": project_info.selected_components,
            "parameters": project_info.segmentation_parameters,
        },
        mask_property=mask_property,
    )
