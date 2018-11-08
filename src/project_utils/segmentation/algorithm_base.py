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


class AlgorithmProperty(object):
    """
    :type name: str
    :type value_type: type
    :type default_value: object
    """

    def __init__(self, name, user_name, default_value, options_range, single_steep=None):
        self.name = name
        self.user_name = user_name
        if type(options_range) is list:
            self.value_type = list
        else:
            self.value_type = type(default_value)
        self.default_value = default_value
        self.range = options_range
        self.single_step = single_steep
        if self.value_type is list:
            assert default_value in options_range

    def __repr__(self):
        return f"{self.__class__.__module__}.{self.__class__.__name__}(name='{self.name}', user_name='{self.user_name}', " + \
               f"default_value={self.default_value}, range={self.range})"

class SegmentationAlgorithm(object):
    @classmethod
    def get_fields(cls):
        raise NotImplementedError\

    @classmethod
    def get_name(cls):
        raise NotImplementedError

    def __init__(self):
        super().__init__()
        self.image: Image = None
        self.channel = None
        self.segmentation = None

    def _clean(self):
        self.image = None
        self.segmentation = None

    def calculation_run(self, report_fun):
        raise NotImplementedError()

    def get_info_text(self):
        raise NotImplementedError()

    def get_channel(self, channel_idx):
        return self.image.get_channel(channel_idx)


    def get_gauss(self, gauss_type, gauss_radius):
        if gauss_type == RadiusType.NO:
            return self.channel
        assert isinstance(gauss_type, RadiusType)
        gauss_radius = calculate_operation_radius(gauss_radius, self.image.spacing, gauss_type)
        layer = gauss_type == RadiusType.R2D
        return gaussian(self.channel, gauss_radius, layer=layer)

    def set_image(self, image):
        self.image = image

    def set_exclude_mask(self, exclude_mask):
        """For Stack Seg - designed for mask part of image - maybe use standardize it to mask"""
        pass

    def set_parameters(self, *args, **kwargs):
        raise NotImplementedError()
