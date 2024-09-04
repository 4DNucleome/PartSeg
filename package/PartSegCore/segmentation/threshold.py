import typing
import warnings
from abc import ABC

import numpy as np
import SimpleITK as sitk
from local_migrator import register_class, rename_key, update_argument
from pydantic import Field

from PartSegCore.algorithm_describe_base import AlgorithmDescribeBase, AlgorithmSelection
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegCore.segmentation.utils import close_small_holes
from PartSegCore.utils import BaseModel


class SingleThresholdParams(BaseModel):
    threshold: float = Field(8000.0, ge=-100000, le=100000, title="Threshold", description="Threshold values")


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("masked", "apply_mask"))])
class SimpleITKThresholdParams128(BaseModel):
    apply_mask: bool = Field(True, description="If apply mask before calculate threshold")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("masked", "apply_mask"))])
class SimpleITKThresholdParams256(BaseModel):
    apply_mask: bool = Field(True, description="If apply mask before calculate threshold")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


class MultipleOtsuThresholdParams(BaseModel):
    components: int = Field(2, title="Number of Components", ge=2, lt=100)
    border_component: int = Field(1, title="Border Component", ge=1, lt=100)
    valley: bool = Field(True, title="Valley emphasis")
    bins: int = Field(128, title="Number of histogram bins", ge=8, le=2**16)


class BaseThreshold(AlgorithmDescribeBase, ABC):
    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: BaseModel,
        operator: typing.Callable[[object, object], bool],
    ):
        raise NotImplementedError


class ManualThreshold(BaseThreshold):
    __argument_class__ = SingleThresholdParams

    @classmethod
    def get_name(cls):
        return "Manual"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: SingleThresholdParams, operator
    ):
        result = np.array(operator(data, arguments.threshold)).astype(np.uint8)
        if mask is not None:
            result[mask == 0] = 0
        return result, arguments.threshold


class SitkThreshold(BaseThreshold, ABC):
    __argument_class__ = SimpleITKThresholdParams128

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: SimpleITKThresholdParams128, operator
    ):
        if mask is not None and mask.dtype != np.uint8 and arguments.apply_mask:
            mask = (mask > 0).astype(np.uint8)
        ob, bg, th_op = (0, 1, np.min) if operator(1, 0) else (1, 0, np.max)
        image_sitk = sitk.GetImageFromArray(data)
        if arguments.apply_mask and mask is not None:
            mask_sitk = sitk.GetImageFromArray(mask)
            calculated = cls.calculate_threshold(image_sitk, mask_sitk, ob, bg, arguments.bins, True, 1)
        else:
            calculated = cls.calculate_threshold(image_sitk, ob, bg, arguments.bins)
        result = sitk.GetArrayFromImage(calculated)
        if mask is not None:
            result[mask == 0] = 0
        threshold = th_op(data[result > 0]) if np.any(result) else th_op(-data)
        return result, threshold

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        raise NotImplementedError


class OtsuThreshold(SitkThreshold):
    @classmethod
    def get_name(cls):
        return "Otsu"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.OtsuThreshold(*args)


class LiThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Li"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.LiThreshold(*args)


class MaximumEntropyThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Maximum Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MaximumEntropyThreshold(*args)


class RenyiEntropyThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Renyi Entropy"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.RenyiEntropyThreshold(*args)


class ShanbhagThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Shanbhag"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.ShanbhagThreshold(*args)


class TriangleThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Triangle"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.TriangleThreshold(*args)


class YenThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Yen"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.YenThreshold(*args)


class HuangThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Huang"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.HuangThreshold(*args)


class IntermodesThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Intermodes"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        try:
            return sitk.IntermodesThreshold(*args)
        except RuntimeError as e:
            if "Exceeded maximum iterations for histogram smoothing" in e.args[0]:
                raise SegmentationLimitException(*e.args) from e
            raise


class IsoDataThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Iso Data"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.IsoDataThreshold(*args)


class KittlerIllingworthThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Kittler Illingworth"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        try:
            return sitk.KittlerIllingworthThreshold(*args)
        except RuntimeError as e:
            if "sigma2 <= 0" in e.args[0]:
                raise SegmentationLimitException(*e.args) from e
            raise


class MomentsThreshold(SitkThreshold):
    __argument_class__ = SimpleITKThresholdParams256

    @classmethod
    def get_name(cls):
        return "Moments"

    @staticmethod
    def calculate_threshold(*args, **kwargs):
        return sitk.MomentsThreshold(*args)


class MultipleOtsuThreshold(BaseThreshold):
    __argument_class__ = MultipleOtsuThresholdParams

    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: MultipleOtsuThresholdParams,
        operator: typing.Callable[[object, object], bool],
    ):
        cleaned_image_sitk = sitk.GetImageFromArray(data)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, arguments.components, 0, arguments.bins, arguments.valley)
        res = sitk.GetArrayFromImage(res)
        if operator(1, 0):
            res = (res >= arguments.border_component).astype(np.uint8)
            threshold = np.min(data[res > 0]) if np.any(res) else np.max(data)
        else:
            res = (res < arguments.border_component).astype(np.uint8)
            threshold = np.max(data[res > 0]) if np.any(res) else np.min(data)
        if mask is not None:
            res[mask == 0] = 0
        return res, threshold

    @classmethod
    def get_name(cls) -> str:
        return "Multiple Otsu"


class ThresholdSelection(AlgorithmSelection, class_methods=["calculate_mask"], suggested_base_class=BaseThreshold):
    pass


ThresholdSelection.register(ManualThreshold)
ThresholdSelection.register(OtsuThreshold)
ThresholdSelection.register(LiThreshold)
ThresholdSelection.register(RenyiEntropyThreshold)
ThresholdSelection.register(ShanbhagThreshold)
ThresholdSelection.register(TriangleThreshold)
ThresholdSelection.register(YenThreshold)
ThresholdSelection.register(HuangThreshold)
ThresholdSelection.register(IntermodesThreshold)
ThresholdSelection.register(IsoDataThreshold)
ThresholdSelection.register(KittlerIllingworthThreshold)
ThresholdSelection.register(MomentsThreshold)
ThresholdSelection.register(MaximumEntropyThreshold)
ThresholdSelection.register(MultipleOtsuThreshold)


class DoubleThresholdParams(BaseModel):
    core_threshold: ThresholdSelection = ThresholdSelection.get_default()
    base_threshold: ThresholdSelection = ThresholdSelection.get_default()


class DoubleThreshold(BaseThreshold):
    __argument_class__ = DoubleThresholdParams

    @classmethod
    def get_name(cls):
        return "Base/Core"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls, data: np.ndarray, mask: typing.Optional[np.ndarray], arguments: DoubleThresholdParams, operator
    ):
        thr: BaseThreshold = ThresholdSelection[arguments.core_threshold.name]
        mask1, thr_val1 = thr.calculate_mask(data, mask, arguments.core_threshold.values, operator)

        thr: BaseThreshold = ThresholdSelection[arguments.base_threshold.name]
        mask2, thr_val2 = thr.calculate_mask(data, mask, arguments.base_threshold.values, operator)
        mask2[mask2 > 0] = 1
        mask2[mask1 > 0] = 2
        return mask2, (thr_val1, thr_val2)


class RangeThresholdParams(DoubleThresholdParams):
    core_threshold: ThresholdSelection = Field(default_factory=ThresholdSelection.get_default, title="Upper threshold")
    base_threshold: ThresholdSelection = Field(default_factory=ThresholdSelection.get_default, title="Lower threshold")


class RangeThreshold(DoubleThreshold):
    __argument_class__ = RangeThresholdParams

    @classmethod
    def get_name(cls):
        return "Range"


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("hist_num", "bins"))])
class DoubleOtsuParams(BaseModel):
    valley: bool = Field(True, title="Valley emphasis")
    bins: int = Field(128, title="Histogram bins", ge=8, le=2**16)


class DoubleOtsu(BaseThreshold):
    __argument_class__ = DoubleOtsuParams

    @classmethod
    def get_name(cls):
        return "Double Otsu"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: DoubleOtsuParams,
        operator: typing.Callable[[object, object], bool],
    ):
        cleaned_image_sitk = sitk.GetImageFromArray(data)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, 2, 0, arguments.bins, arguments.valley)
        res = sitk.GetArrayFromImage(res)
        if not np.any(res.flat):
            return res, (0, 0)
        thr1 = data[res == 2].min()
        thr2 = data[res == 1].min()
        return res, (thr1, thr2)


class MultipleOtsuDoubleThresholdParams(BaseModel):
    components: int = Field(2, title="Number of Components", ge=2, lt=100)
    lower_component: int = Field(1, title="Lower Component", ge=1, lt=100)
    upper_component: int = Field(1, title="Upper Component", ge=1, lt=100)
    valley: bool = Field(True, title="Valley emphasis")
    bins: int = Field(128, title="Number of histogram bins", ge=8, le=2**16)


class MultipleOtsu(BaseThreshold):
    __argument_class__ = MultipleOtsuDoubleThresholdParams

    @classmethod
    def get_name(cls):
        return "Multiple Otsu"

    @classmethod
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: MultipleOtsuDoubleThresholdParams,
        operator: typing.Callable[[object, object], bool],
    ):
        cleaned_image_sitk = sitk.GetImageFromArray(data)
        res = sitk.OtsuMultipleThresholds(cleaned_image_sitk, arguments.components, 0, arguments.bins, arguments.valley)
        res = sitk.GetArrayFromImage(res)
        if not np.any(res.flat):
            return res, (0, 0)
        map_component = np.zeros(arguments.components + 1, dtype=np.uint8)
        map_component[: arguments.lower_component] = 0
        map_component[arguments.lower_component : arguments.upper_component] = 1
        map_component[arguments.upper_component :] = 2
        res2 = map_component[res]
        thr1 = data[res2 == 2].min() if np.any(res2 == 2) else data[res2 == 1].max()
        thr2 = data[res2 == 1].min() if np.any(res2 == 1) else data.max()
        return res2, (thr1, thr2)


@register_class(version="0.0.1", migrations=[("0.0.1", rename_key("minimum_radius", "minimum_border_distance"))])
class MaximumDistanceWatershedParams(BaseModel):
    threshold: ThresholdSelection = ThresholdSelection.get_default()
    dilate_radius: int = Field(5, title="Dilate Radius", ge=1, le=100, description="To merge small objects")
    minimum_size: int = Field(100, title="Minimum Size", ge=1, le=1000000, description="To remove small objects")
    minimum_border_distance: int = Field(
        10,
        title="Border Radius",
        ge=0,
        le=100,
        description="Minimum distance of local maxima from the border. To avoid artifacts",
    )


class MaximumDistanceCore(BaseThreshold):
    """
    This Is algorithm intended to bue used in "* threshold with watershed" algorithms.
    It generates array with three values:

     * 0 - background,
     * 1 - area to watershed,
     * 2 - core objects to start watershed from

    This algorithm is developed to make possible split of almost convex objects that are touching each other.
    Core objects are identified as local maxima of distance from the border.

    To perform this task the following steps are performed:

    1. Thresholding - to detect whole area of objects. This is controlled by ``threshold`` parameter.
    2. Remove small objects - to remove small objects. This is controlled by ``minimum_size` parameter.
    3. Small objects close - to merge small objects. As distance transform is used, it is required small holes.
       This steep closes holes smaller tan 10px.
    4. Distance transform - to find distance from the border
    5. Identify local maxima - to find core objects
    6. Remove local maxima that are too close to the border - to avoid artifacts.
       This distance is controlled by ``minimum_border_distance`` parameter.
    7. Dilate core objects - to make them bigger. For elongated objects it is possible to have multiple local
       maxima along longest axis of object. This step is to merge them.
       This distance is controlled by ``dilate_radius`` parameter.


    This is algorithm that detect core objects
    """

    __argument_class__ = MaximumDistanceWatershedParams

    @classmethod
    def get_name(cls):
        return "Maximum Distance Core"

    @classmethod
    @update_argument("arguments")
    def calculate_mask(
        cls,
        data: np.ndarray,
        mask: typing.Optional[np.ndarray],
        arguments: MaximumDistanceWatershedParams,
        operator: typing.Callable[[object, object], bool],
    ):
        thr: BaseThreshold = ThresholdSelection[arguments.threshold.name]
        mask1, thr_val = thr.calculate_mask(data, mask, arguments.threshold.values, operator)
        mask1 = sitk.GetArrayFromImage(
            sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(mask1)), arguments.minimum_size)
        )
        if not np.any(mask1):
            return mask1, (thr_val, thr_val)
        mask2 = close_small_holes((mask1 > 0), 10)
        data = sitk.GetArrayFromImage(sitk.DanielssonDistanceMap(sitk.GetImageFromArray((mask2 == 0).astype(np.uint8))))

        maxima = sitk.GetArrayFromImage(sitk.RegionalMaxima(sitk.GetImageFromArray(data)))
        maxima[data < arguments.minimum_border_distance] = 0

        dilated_maxima = sitk.GetArrayFromImage(
            sitk.BinaryDilate(sitk.GetImageFromArray(maxima), [arguments.dilate_radius] * 3)
        )

        mask1[mask1 > 0] = 1
        mask1[dilated_maxima > 0] = 2
        if operator(0, 1):
            meth = np.amax
        else:
            meth = np.amin
        return mask1, (thr_val, float(meth(data[mask1 == 2]) if np.any(mask1 == 2) else meth(data[mask1 == 1])))


class DoubleThresholdSelection(
    AlgorithmSelection, class_methods=["calculate_mask"], suggested_base_class=BaseThreshold
):
    pass


DoubleThresholdSelection.register(DoubleThreshold)
DoubleThresholdSelection.register(DoubleOtsu)
DoubleThresholdSelection.register(MultipleOtsu)
DoubleThresholdSelection.register(MaximumDistanceCore, old_names=["Maximum Distance Watershed"])


class RangeThresholdSelection(AlgorithmSelection, class_methods=["calculate_mask"], suggested_base_class=BaseThreshold):
    pass


RangeThresholdSelection.register(RangeThreshold)
RangeThresholdSelection.register(DoubleOtsu)


def __getattr__(name):  # pragma: no cover
    if name == "threshold_dict":
        warnings.warn(
            "threshold_dict is deprecated. Please use ThresholdSelection instead", category=FutureWarning, stacklevel=2
        )
        return ThresholdSelection.__register__

    if name == "double_threshold_dict":
        warnings.warn(
            "double_threshold_dict is deprecated. Please use DoubleThresholdSelection instead",
            category=FutureWarning,
            stacklevel=2,
        )
        return DoubleThresholdSelection.__register__

    raise AttributeError(f"module {__name__} has no attribute {name}")
