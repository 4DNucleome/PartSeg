"""
This module contains some not fully intuitive algorithm for splitting masked area. 
They are designed for measurements, but should be also available as segmentation algorithm to
allow user to preview, for better intuition how it works and which parameters are good for their purpose.

Better option is to implement this utils as class based with base class :py:class:`AlgorithmDescribeBase`.
Then do not need to manage algorithm parameters in places where it is used.

Both class from this module are designed for spherical mask, but may be useful als for others.
"""
import typing

import SimpleITK
import numpy as np
from scipy.ndimage import distance_transform_edt

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty
from .universal_const import UNIT_SCALE, Units


class BorderRim(AlgorithmDescribeBase):
    """
    This class implement border rim (Annulus like) calculation.
    For given mask and image spacing it marks as 1 all pixels in given distance from mask border.

    https://en.wikipedia.org/wiki/Annulus_(mathematics)
    """
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 10000), property_type=float),
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units)]

    @classmethod
    def get_name(cls) -> str:
        return "Border Rim"

    @staticmethod
    def border_mask(mask: np.ndarray, distance: float, units: Units, voxel_size, **_) -> typing.Optional[np.ndarray]:
        """
        This is function which implement calculation.

        :param mask: area for which rim should be calculated. 2d or 3d numpy array,
        :param distance: distance from border which will be marked.
        :param units: in which unit distance is given
        :param voxel_size: Image spacing in absolute units
        :param _: ignored arguments
        :return: border rim marked with 1
        """
        if mask is None:
            return None
        units_scalar = UNIT_SCALE[units.value]
        final_radius = [int((distance / units_scalar) / x) for x in reversed(voxel_size)]
        mask = np.array(mask > 0)
        mask = mask.astype(np.uint8)
        eroded = SimpleITK.GetArrayFromImage(SimpleITK.BinaryErode(SimpleITK.GetImageFromArray(mask.squeeze()),
                                                                   final_radius))
        eroded = eroded.reshape(mask.shape)
        mask[eroded > 0] = 0
        return mask


class SplitMaskOnPart(AlgorithmDescribeBase):
    """
    This class contains implementation of splitting mask on parts based on distance from borders.
    Has two modes or working. It may split on parts with same thickness or same volume..
    """
    @classmethod
    def get_name(cls) -> str:
        return "Mask on Part"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [AlgorithmProperty("num_of_parts", "Number of Part", 2, (1, 1024)),
                AlgorithmProperty("equal_volume", "Equal Volume", False)]

    @staticmethod
    def split(mask: np.ndarray, num_of_parts: int, equal_volume: bool, voxel_size):
        """
        This is function which implement calculation.

        :param mask: area for which rim should be calculated. 2d or 3d numpy array
        :param num_of_parts: num of parts on which mask should be split
        :param equal_volume: if split should be on equal volume or equal thick
        :param voxel_size: image voxel size
        :return: mask region labelled starting from 1 near border
        """
        distance_arr = distance_transform_edt(mask, sampling=voxel_size)
        if equal_volume:
            hist, bins = np.histogram(distance_arr[distance_arr > 0], bins=10*num_of_parts)
            total = np.sum(hist)
            levels, step = np.linspace(0, total, num_of_parts+1, True, retstep=True)
            bounds = [0]
            i = 1
            cum_sum = 0
            for val, begin, end in zip(hist, bins, bins[1:]):
                cum_sum = cum_sum + val
                if cum_sum > levels[i]:
                    exceed = (cum_sum - levels[i])/step
                    bounds.append(begin + (end - begin)*exceed)
                    i += 1
        else:
            max_dist = np.max(distance_arr)
            bounds = np.linspace(0, max_dist, num_of_parts, False)
        mask = np.zeros(mask.shape, dtype=np.uint8 if num_of_parts < 255 else np.uint16)
        for bound in bounds:
            mask[distance_arr > bound] += 1
        return mask
