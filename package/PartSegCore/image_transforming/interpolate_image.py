from typing import Callable, List, Optional, Tuple, Union

from scipy.ndimage import zoom

from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.image_transforming.transform_base import TransformBase
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image


class InterpolateImage(TransformBase):
    @classmethod
    def get_fields(cls):
        return ["It can be very slow.", AlgorithmProperty("scale", "Scale", 1.0)]

    @classmethod
    def get_fields_per_dimension(cls, image: Image) -> List[Union[str, AlgorithmProperty]]:
        component_list = list(image.get_dimension_letters())
        return [
            "it can be very slow",
            *[AlgorithmProperty(f"scale_{i.lower()}", f"Scale {i}", 1.0) for i in reversed(component_list)],
        ]

    @classmethod
    def get_name(cls):
        return "Spline Interpolate"

    @classmethod
    def transform(
        cls,
        image: Image,
        roi_info: Optional[ROIInfo],
        arguments: dict,
        callback_function: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Image, Optional[ROIInfo]]:
        keys = [x for x in arguments if x.startswith("scale")]
        keys_order = Image.axis_order.lower()
        scale_factor = [1.0] * len(keys_order)
        if len(keys) == 1 and keys[0] == "scale":
            for letter in image.get_dimension_letters().lower():
                scale_factor[keys_order.index(letter)] = arguments["scale"]
            spacing = [x / arguments["scale"] for x in image.spacing]
        else:
            # assume that all keys are in format scale_{}
            for key in keys:
                letter = key[-1]
                scale_factor[keys_order.index(letter)] = arguments[key]
            spacing = [
                x / arguments[f"scale_{y}"] for x, y in zip(image.spacing, image.get_dimension_letters().lower())
            ]
        array = zoom(image.get_data(), scale_factor, mode="mirror")
        if image.mask is not None:
            mask = zoom(image.mask, scale_factor[:-1], mode="mirror")
        else:
            mask = None
        return image.substitute(data=array, image_spacing=spacing, mask=mask), None

    @classmethod
    def calculate_initial(cls, image: Image):
        min_val = min(image.spacing)
        return {
            f"scale_{letter}": x / min_val for x, letter in zip(image.spacing, image.get_dimension_letters().lower())
        }
