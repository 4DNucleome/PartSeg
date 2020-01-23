from .channel_class import Channel
from .class_generator import BaseSerializableClass
from .image_operations import RadiusType


class CmapProfileBase:
    channel: Channel
    gauss_type: RadiusType
    gauss_radius: float
    center_data: bool
    rotation_axis: bool
    cut_obsolete_area: bool


class CmapProfile(CmapProfileBase, BaseSerializableClass):
    pass
