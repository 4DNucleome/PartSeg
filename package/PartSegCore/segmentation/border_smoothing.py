import warnings
from abc import ABC

import numpy as np
import SimpleITK as sitk
from local_migrator import update_argument
from pydantic import Field

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmSelection
from PartSegCore.segmentation.watershed import NeighType, get_neighbourhood
from PartSegCore.utils import BaseModel


class BaseSmoothing(AlgorithmDescribeBase, ABC):
    __argument_class__ = BaseModel

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        raise NotImplementedError


class NoneSmoothing(BaseSmoothing):
    @classmethod
    def get_name(cls) -> str:
        return "None"

    @classmethod
    def smooth(cls, segmentation: np.ndarray, arguments: dict) -> np.ndarray:
        return segmentation


class OpeningSmoothingParams(BaseModel):
    smooth_border_radius: int = Field(2, title="Smooth borders radius", ge=1, le=20)


class OpeningSmoothing(BaseSmoothing):
    __argument_class__ = OpeningSmoothingParams

    @classmethod
    def get_name(cls) -> str:
        return "Opening"

    @classmethod
    @update_argument("arguments")
    def smooth(cls, segmentation: np.ndarray, arguments: OpeningSmoothingParams) -> np.ndarray:
        radius = arguments.smooth_border_radius
        if isinstance(radius, (int, float)):
            radius = [radius] * segmentation.ndim
        return sitk.GetArrayFromImage(sitk.BinaryMorphologicalOpening(sitk.GetImageFromArray(segmentation), radius))


class VoteSmoothingParams(BaseModel):
    neighbourhood_type: NeighType = Field(
        NeighType.edges, title="Side Neighbourhood", description="use 6, 18 or 26 neighbourhood (5, 8, 8 for 2d data)"
    )
    support_level: int = Field(
        1,
        title="Support level",
        ge=1,
        le=27,
        description="How many voxels in neighbourhood need to be labeled to preserve pixel",
    )


class VoteSmoothing(BaseSmoothing):
    __argument_class__ = VoteSmoothingParams

    @classmethod
    def get_name(cls) -> str:
        return "Vote"

    @classmethod
    @update_argument("arguments")
    def smooth(cls, segmentation: np.ndarray, arguments: VoteSmoothingParams) -> np.ndarray:
        segmentation_bin = (segmentation > 0).astype(np.uint8)
        count_array = np.zeros(segmentation_bin.shape, dtype=segmentation.dtype)
        neighbourhood = get_neighbourhood(segmentation_bin.squeeze().shape, arguments.neighbourhood_type)
        axis = tuple(range(len(segmentation_bin.shape)))
        for shift in neighbourhood:
            count_array += np.roll(segmentation_bin, shift[: segmentation_bin.ndim], axis)
        segmentation = segmentation.copy()
        count_array = count_array.reshape(segmentation.shape)
        segmentation[count_array < arguments.support_level] = 0
        return segmentation


class IterativeSmoothingParams(VoteSmoothingParams):
    max_steps: int = Field(
        1, title="Max steps", description="How many voxels in neighbourhood need to be labeled to preserve pixel"
    )


class IterativeVoteSmoothing(BaseSmoothing):
    __argument_class__ = IterativeSmoothingParams

    @classmethod
    def get_name(cls) -> str:
        return "Iterative Vote"

    @classmethod
    @update_argument("arguments")
    def smooth(cls, segmentation: np.ndarray, arguments: IterativeSmoothingParams) -> np.ndarray:
        segmentation_bin = (segmentation > 0).astype(np.uint8)
        count_array = np.zeros(segmentation_bin.shape, dtype=segmentation.dtype)
        neighbourhood = get_neighbourhood(segmentation_bin.squeeze().shape, arguments.neighbourhood_type)
        segmentation = segmentation.copy()
        count_point = np.count_nonzero(segmentation)
        axis = tuple(range(len(segmentation_bin.shape)))
        for _ in range(arguments.max_steps):
            for shift in neighbourhood:
                count_array += np.roll(segmentation_bin, shift[: segmentation_bin.ndim], axis)
            segmentation_bin[count_array < arguments.support_level] = 0
            count_point2 = np.count_nonzero(segmentation_bin)
            if count_point2 == count_point:
                break
            count_point = count_point2
            count_array[:] = 0
        segmentation[segmentation_bin.reshape(segmentation.shape) == 0] = 0
        return segmentation


class SmoothAlgorithmSelection(AlgorithmSelection, class_methods=["smooth"], suggested_base_class=BaseSmoothing):
    pass


SmoothAlgorithmSelection.register(NoneSmoothing)
SmoothAlgorithmSelection.register(OpeningSmoothing)
SmoothAlgorithmSelection.register(VoteSmoothing)
SmoothAlgorithmSelection.register(IterativeVoteSmoothing)


def __getattr__(name):  # pragma: no cover
    if name == "smooth_dict":
        warnings.warn(
            "threshold_dict is deprecated. Please use SmoothAlgorithmSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return SmoothAlgorithmSelection.__register__

    raise AttributeError(f"module {__name__} has no attribute {name}")
