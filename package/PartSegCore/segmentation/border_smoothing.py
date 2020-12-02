from abc import ABC

import numpy as np
import SimpleITK as sitk

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmProperty, Register
from PartSegCore.segmentation.watershed import NeighType, get_neighbourhood


class BaseSmoothing(AlgorithmDescribeBase, ABC):
    @classmethod
    def get_fields(cls):
        return []

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        raise NotImplementedError()


class NoneSmoothing(BaseSmoothing):
    @classmethod
    def get_name(cls) -> str:
        return "None"

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        return segmentation


class OpeningSmoothing(BaseSmoothing):
    @classmethod
    def get_name(cls) -> str:
        return "Opening"

    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty("smooth_border_radius", "Smooth borders radius", 2, (1, 20), 1)]

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        radius = arguments["smooth_border_radius"]
        if isinstance(radius, (int, float)):
            radius = [radius] * segmentation.ndim
        return sitk.GetArrayFromImage(sitk.BinaryMorphologicalOpening(sitk.GetImageFromArray(segmentation), radius))


class VoteSmoothing(BaseSmoothing):
    @classmethod
    def get_name(cls) -> str:
        return "Vote"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "neighbourhood_type",
                "Side Neighbourhood",
                NeighType.edges,
                help_text="use 6, 18 or 26 neighbourhood (5, 8, 8 for 2d data)",
            ),
            AlgorithmProperty(
                "support_level",
                "Support level",
                1,
                (1, 27),
                1,
                help_text="How many voxels in neighbourhood need to be labeled to preserve pixel",
            ),
        ]

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        segmentation_bin = (segmentation > 0).astype(np.uint8)
        count_array = np.zeros(segmentation_bin.shape, dtype=segmentation.dtype)
        neighbourhood = get_neighbourhood(segmentation_bin.squeeze().shape, arguments["neighbourhood_type"])
        axis = tuple(range(len(segmentation_bin.shape)))
        for shift in neighbourhood:
            count_array += np.roll(segmentation_bin, shift, axis)
        segmentation = segmentation.copy()
        count_array = count_array.reshape(segmentation.shape)
        segmentation[count_array < arguments["support_level"]] = 0
        return segmentation


class IterativeVoteSmoothing(BaseSmoothing):
    @classmethod
    def get_name(cls) -> str:
        return "Iterative Vote"

    @classmethod
    def get_fields(cls):
        return [
            AlgorithmProperty(
                "neighbourhood_type", "Side Neighbourhood", NeighType.edges, help_text="use 6, 18 or 26 neighbourhood"
            ),
            AlgorithmProperty(
                "support_level",
                "Support level",
                1,
                (1, 26),
                1,
                help_text="How many voxels in neighbourhood need to be labeled to preserve pixel",
            ),
            AlgorithmProperty(
                "max_steps",
                "Max steps",
                1,
                (1, 100),
                1,
                help_text="How many voxels in neighbourhood need to be labeled to preserve pixel",
            ),
        ]

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        segmentation_bin = (segmentation > 0).astype(np.uint8)
        count_array = np.zeros(segmentation_bin.shape, dtype=segmentation.dtype)
        neighbourhood = get_neighbourhood(segmentation_bin.squeeze().shape, arguments["neighbourhood_type"])
        segmentation = segmentation.copy()
        count_point = np.count_nonzero(segmentation)
        axis = tuple(range(len(segmentation_bin.shape)))
        for _ in range(arguments["max_steps"]):
            for shift in neighbourhood:
                count_array += np.roll(segmentation_bin, shift, axis)
            segmentation_bin[count_array < arguments["support_level"]] = 0
            count_point2 = np.count_nonzero(segmentation_bin)
            if count_point2 == count_point:
                break
            count_point = count_point2
            count_array[:] = 0
        segmentation[segmentation_bin.reshape(segmentation.shape) == 0] = 0
        return segmentation


smooth_dict = Register(NoneSmoothing, OpeningSmoothing, VoteSmoothing, IterativeVoteSmoothing)
