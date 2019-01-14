from partseg_utils.channel_class import Channel
from partseg_utils.class_generator import BaseReadonlyClass
from partseg_utils.image_operations import RadiusType


class CmapProfileBase:
    channel: Channel
    gauss_type: RadiusType
    gauss_radius: float
    center_data: bool
    rotation_axis: bool
    cut_obsolete_area: bool


class CmapProfile(CmapProfileBase, BaseReadonlyClass):
    pass