from project_utils.image_operations import gaussian, RadiusType
from tiff_image import Image


def calculate_operation_radius(radius, spacing, gauss_type):
    if gauss_type == RadiusType.R2D:
        if len(spacing) == 3:
            spacing = spacing[1:]
    base = min(spacing)
    if base != max(spacing):
        ratio = [x / base for x in  spacing]
        return [radius / r for r in ratio]
    return  radius


class SegmentationAlgorithm(object):
    def __init__(self):
        super().__init__()
        self.image: Image = None
        self.channel = None
        self.segmentation = None
        self.spacing = None
        self.use_psychical_unit = False
        self.unit_scalar = 1

    def _clean(self):
        self.image = None
        self.segmentation = None

    def calculation_run(self, report_fun):
        raise NotImplementedError()

    def get_info_text(self):
        raise NotImplementedError()

    def set_size_information(self, spacing, use_physical_unit, unit_scalar):
        self.unit_scalar = unit_scalar
        self.spacing = spacing
        self.use_psychical_unit = use_physical_unit

    def get_channel(self, channel_idx):
        return self.image.get_channel(channel_idx)


    def get_gauss(self, gauss_type, gauss_radius):
        if gauss_type == RadiusType.NO:
            return self.channel
        assert isinstance(gauss_type, RadiusType)
        gauss_radius = calculate_operation_radius(gauss_radius, self.spacing, gauss_type)
        layer = gauss_type == RadiusType.R2D
        return gaussian(self.channel, gauss_radius, layer=layer)

    def set_image(self, image):
        self.image = image

    def set_exclude_mask(self):
        """For Stack Seg - designed for mask part of image - maybe use standardize it to mask"""
        pass

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
