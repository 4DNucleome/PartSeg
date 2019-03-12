from typing import List, Callable, Optional
from scipy.ndimage import zoom

from PartSeg.tiff_image import Image
from PartSeg.utils.algorithm_describe_base import AlgorithmProperty
from .transform_base import TransformBase


class InterpolateImage(TransformBase):
    @classmethod
    def get_fields(cls):
        return [AlgorithmProperty(f"scale", "Scale", 1.0)]

    @classmethod
    def get_fields_per_dimension(cls, component_list: List[str]):
        return [AlgorithmProperty(f"scale_{l}", f"Scale {l}", 1.0) for l in reversed(component_list)]

    @classmethod
    def get_name(cls):
        return "Spline Interpolate"

    @classmethod
    def transform(cls, image: Image, arguments: dict, callback_function: Optional[Callable[[str, int], None]] = None) \
            -> Image:
        keys = [x for x in arguments.keys() if x.startswith("scale")]
        keys_order = Image.return_order.lower()
        scale_factor = [1.0] * len(keys_order)
        if len(keys) == 1 and keys[0] == "scale":
            for letter in image.get_dimension_letters():
                scale_factor[keys_order.index(letter)] = arguments["scale"]
            spacing = [x/arguments["scale"] for x in image.spacing]
        else:
            # assume that all keys are in format scale_{}
            for key in keys:
                letter = key[-1]
                scale_factor[keys_order.index(letter)] = arguments[key]
            spacing = [x/arguments[f"scale_{y}"] for x, y in zip(image.spacing, image.get_dimension_letters())]
        array = zoom(image.get_data(), scale_factor, mode="mirror")
        if image.mask is not None:
            mask = zoom(image.mask, scale_factor[:-1], mode="mirror")
        else:
            mask = None
        return image.substitute(data=array, image_spacing=spacing, mask=mask)

    @classmethod
    def calculate_initial(cls, image: Image):
        min_val = min(image.spacing)
        return dict([(f"scale_{l}", x/min_val) for x, l in zip(image.spacing, image.get_dimension_letters())])
