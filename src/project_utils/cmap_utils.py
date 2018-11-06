from project_utils.class_generator import BaseReadonlyClass
from project_utils.image_operations import RadiusType


class CmapProfileBase:
    gauss_type: RadiusType
    center_data: bool
    rotation_axis: bool
    cut_obsolete_area: bool


class CmapProfile(BaseReadonlyClass):
    pass