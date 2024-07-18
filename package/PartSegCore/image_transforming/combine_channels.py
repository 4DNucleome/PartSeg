from enum import Enum, auto
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.image_transforming.transform_base import TransformBase
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class CombineMode(Enum):
    Max = auto()
    Sum = auto()


class CombineChannels(TransformBase):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("combine_mode", "Combine Mode", CombineMode.Sum)]

    @classmethod
    def get_fields_per_dimension(cls, image: Image) -> List[Union[str, AlgorithmProperty]]:
        return [
            AlgorithmProperty("combine_mode", "Combine Mode", CombineMode.Sum),
            *[AlgorithmProperty(f"channel_{i}", f"Channel {i}", False) for i in range(image.channels)],
        ]

    @classmethod
    def get_name(cls):
        return "Combine channels"

    @classmethod
    def transform(
        cls,
        image: Image,
        roi_info: Optional[ROIInfo],
        arguments: dict,
        callback_function: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Image, Optional[ROIInfo]]:
        channels = [i for i, x in enumerate(x for x in arguments.items() if x[0].startswith("channel")) if x[1]]
        if not channels:
            return image, roi_info
        channel_array = [image.get_channel(i) for i in channels]
        if len(channels) == 1:
            new_channel = channel_array[0]
        elif arguments["combine_mode"] == CombineMode.Max:
            new_channel = np.max(channel_array, axis=0)
        else:
            new_channel = np.sum(channel_array, axis=0)
        all_channels = [image.get_channel(i) for i in range(image.channels)]
        all_channels.append(new_channel)
        channel_names = [*image.channel_names, "combined"]
        contrast_limits = [*image.get_ranges(), (np.min(new_channel), np.max(new_channel))]
        return image.substitute(data=all_channels, channel_names=channel_names, ranges=contrast_limits), roi_info

    @classmethod
    def calculate_initial(cls, image: Image):
        min_val = min(image.spacing)
        return {
            f"scale_{letter}": x / min_val for x, letter in zip(image.spacing, image.get_dimension_letters().lower())
        }
