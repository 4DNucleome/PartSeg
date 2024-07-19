from enum import Enum
from typing import Callable, Optional, Tuple

import numpy as np
from pydantic import Field

from PartSegCore.image_transforming.transform_base import TransformBase
from PartSegCore.roi_info import ROIInfo
from PartSegCore.utils import BaseModel
from PartSegImage import Image


class ProjectionType(Enum):
    MAX = "max"
    MIN = "min"
    MEAN = "mean"
    SUM = "sum"


class ImageProjectionParams(BaseModel):
    projection_type: ProjectionType = Field(ProjectionType.MAX, suffix="Mask and ROI projection will always use max")
    keep_mask: bool = False
    keep_roi: bool = False


def _calc_target_shape(image: Image):
    new_shape = list(image.shape)
    new_shape[image.array_axis_order.index("Z")] = 1
    return tuple(new_shape)


class ImageProjection(TransformBase):
    __argument_class__ = ImageProjectionParams

    @classmethod
    def transform(
        cls,
        image: Image,
        roi_info: ROIInfo,
        arguments: ImageProjectionParams,  # type: ignore[override]
        callback_function: Optional[Callable[[str, int], None]] = None,
    ) -> Tuple[Image, Optional[ROIInfo]]:
        project_operator = getattr(np, arguments.projection_type.value)
        axis = image.array_axis_order.index("Z")
        target_shape = _calc_target_shape(image)
        spacing = list(image.spacing)
        spacing.pop(axis - 1 if image.time_pos < image.stack_pos else axis)
        new_channels = [
            project_operator(image.get_channel(i), axis=axis).reshape(target_shape) for i in range(image.channels)
        ]
        new_mask = None
        if arguments.keep_mask and image.mask is not None:
            new_mask = np.max(image.mask, axis=axis).reshape(target_shape)

        roi = None
        if arguments.keep_roi and roi_info.roi is not None:
            roi = ROIInfo(np.max(image.fit_array_to_image(roi_info.roi), axis=axis).reshape(target_shape))
        return (
            image.__class__(
                data=new_channels,
                image_spacing=tuple(spacing),
                channel_names=image.channel_names,
                mask=new_mask,
                axes_order=image.axis_order,
            ),
            roi,
        )

    @classmethod
    def get_fields_per_dimension(cls, image: Image):
        return cls.__argument_class__

    @classmethod
    def calculate_initial(cls, image: Image):
        return cls.get_default_values()

    @classmethod
    def get_name(cls) -> str:
        return "Image Projection"
