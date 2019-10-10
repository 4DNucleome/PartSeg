import typing

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty
from .universal_const import UNIT_SCALE, Units


class BorderRim(AlgorithmDescribeBase):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("distance", "Distance", 0.0, options_range=(0, 10000), property_type=float),
                AlgorithmProperty("units", "Units", Units.nm, property_type=Units)]

    @classmethod
    def get_name(cls) -> str:
        return "Border Rim"

    @staticmethod
    def border_mask(mask: np.ndarray, distance: float, units: Units, voxel_size, **_) -> typing.Optional[np.ndarray]:
        if mask is None:
            return None
        units_scalar = UNIT_SCALE[units.value]
        final_radius = [int((distance / units_scalar) / x) for x in reversed(voxel_size)]
        mask = np.array(mask > 0)
        mask = mask.astype(np.uint8)
        eroded = sitk.GetArrayFromImage(sitk.BinaryErode(sitk.GetImageFromArray(mask.squeeze()), final_radius))
        eroded = eroded.reshape(mask.shape)
        mask[eroded > 0] = 0
        return mask


class SplitMaskOnPart(AlgorithmDescribeBase):
    @classmethod
    def get_name(cls) -> str:
        return "Mask on Part"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [AlgorithmProperty("num_of_part", "Number of Part", 0),
                AlgorithmProperty("equal_volume", "EqualVolume", False)]

    @staticmethod
    def split(mask: np.ndarray, num_of_parts: int, equal_volume: bool, voxel_size):
        distance_arr = distance_transform_edt(mask, sampling=voxel_size)
        if equal_volume:
            hist, bins = np.histogram(distance_arr[distance_arr > 0], bins=10*num_of_parts)
            total = np.sum(hist)
            levels, step = np.linspace(0, total, num_of_parts, False, retstep=True)
            bounds = [0]
            i = 1
            cum_sum = 0
            for val, begin, end in zip(levels, bins, bins[1:]):
                cum_sum = cum_sum + val
                if cum_sum > levels[i]:
                    exceed = (cum_sum - levels[i])/step
                    bounds.append(begin + (end - begin)*exceed)
                    i += 1
        else:
            max_dist = distance_arr.max()
            bounds = np.linspace(0, max_dist, num_of_parts, False)
        mask = np.zeros(mask.shape, dtype=np.uint8 if num_of_parts < 255 else np.uint16)
        for bound in bounds:
            mask[distance_arr > bound] += 1
        return mask

