import numpy as np
import typing
from .class_generator import BaseSerializableClass
from ..utils.image_operations import dilate, erode, RadiusType
import SimpleITK as sitk

"""MaskProperty = namedtuple("MaskProperty",
                          ["dilate", "dilate_radius", "fill_holes", "max_holes_size", "save_components",
                           "clip_to_mask"])"""


class MaskProperty(BaseSerializableClass):
    """
    :parm dilate: RadiusType ,if need to dilate
    :parm dilate_radius: int,  radius o f dilation calculate with respect of image spacing
    :parm fill_holes: RadiusType
    :parm max_holes_size: int
    :parm save_components: bool
    :parm clip_to_mask: bool
    """
    dilate: RadiusType
    dilate_radius: int
    fill_holes: RadiusType
    max_holes_size: int
    save_components: bool
    clip_to_mask: bool

    def __str__(self):
        return f"Mask property\ndilate: {self.dilate}\n" + \
               (f"dilate radius {self.dilate_radius}\n" if self.dilate != RadiusType.NO else "") + \
               f"fill holes: {self.fill_holes}\n" + \
               (f"max holes size: {self.max_holes_size}\n" if self.fill_holes != RadiusType.NO else "") + \
               f"save components: {self.save_components}\nclip to mask: {self.clip_to_mask}"

def mp_eq(self, other):
    return self.__class__ == other.__class__ and self.dilate == other.dilate and \
           self.fill_holes == other.fill_holes and self.save_components == other.save_components and \
           self.clip_to_mask == other.clip_to_mask and \
           (self.dilate == RadiusType.NO or (self.dilate_radius == other.dilate_radius)) and \
           (self.fill_holes == RadiusType.NO or (self.max_holes_size == other.max_holes_size))

MaskProperty.__eq__ = mp_eq


def calculate_mask(mask_description: MaskProperty, segmentation: np.ndarray, old_mask: typing.Union[None, np.ndarray],
                   spacing: typing.Iterable[typing.Union[float, int]]) -> np.ndarray:
    """
    Function for calculate mask base on MaskProperty.
    If dilate_radius is negative then holes closing is done before erode,
    otherwise it is done after dilate

    :param mask_description: information how calculate mask
    :param segmentation: array on which mask is calculated
    :param old_mask: if in mask_description there is set to crop and old_mask is not None
    then final mask is clipped to this area
    :param spacing: spacing of image. Needed for calculating radius of dilate
    :return: new mask
    """
    spacing_min = min(spacing)
    spacing = [x / spacing_min for x in spacing]
    dilate_radius = [int(abs(mask_description.dilate_radius / x) + 0.5) for x in spacing]
    if mask_description.dilate == RadiusType.R2D:
        dilate_radius = dilate_radius[-2:]
    if mask_description.save_components:
        mask = np.copy(segmentation)
    else:
        mask = np.array(segmentation > 0)

    if mask_description.dilate != RadiusType.NO and mask_description.dilate_radius != 0:
        if mask_description.dilate_radius > 0:
            mask = dilate(mask, dilate_radius, mask_description.dilate == RadiusType.R2D)
            mask = _fill_holes(mask_description, mask)
        elif mask_description.dilate_radius < 0:
            mask = _fill_holes(mask_description, mask)
            mask = erode(mask, dilate_radius, mask_description.dilate == RadiusType.R2D)
    elif mask_description.fill_holes != RadiusType.NO:
        mask = _fill_holes(mask_description, mask)
    if mask_description.clip_to_mask and old_mask is not None:
        mask[old_mask == 0] = 0
    return mask


def cut_components(mask: np.ndarray, image: np.ndarray, borders: int = 0) -> \
        typing.Iterator[typing.Tuple[np.ndarray, typing.List[slice], int]]:
    sizes = np.bincount(mask.flat)
    for i, size in enumerate(sizes[1:], 1):
        if size > 0:
            points = np.nonzero(mask == i)
            lower_bound = np.min(points, axis=1)
            upper_bound = np.max(points, axis=1)
            new_cut = tuple([slice(x, y + 1) for x, y in zip(lower_bound, upper_bound)])
            new_size = [y - x + 1 + 2*borders for x, y in zip(lower_bound, upper_bound)]
            if borders > 0:
                res = np.zeros(new_size, dtype=image.dtype)
                res_cut = tuple([slice(borders, x-borders) for x in res.shape])
                tmp_res = image[new_cut]
                tmp_res[mask[new_cut] != i] = 0
                res[res_cut] = tmp_res
            else:
                res = image[new_cut]
                res[mask[new_cut] != i] = 0
            yield res, tuple(new_cut), i


def _fill_holes(mask_description: MaskProperty, mask: np.ndarray) -> np.ndarray:
    if mask_description.fill_holes == RadiusType.NO:
        return mask
    if mask_description.save_components:
        border = 1
        res_slice = tuple([slice(border, -border) for _ in range(mask.ndim)])
        mask_description_copy = mask_description._replace(save_components=False)
        for component, slice_arr, cmp_num in cut_components(mask, mask, border):
            new_component = _fill_holes(mask_description_copy, component)
            new_component = new_component[res_slice]
            mask[slice_arr][new_component > 0] = cmp_num
    else:
        if mask_description.fill_holes == RadiusType.R2D:
            mask = fill_2d_holes_in_mask(mask, mask_description.max_holes_size)
        elif mask_description.fill_holes == RadiusType.R3D:
            mask = fill_holes_in_mask(mask, mask_description.max_holes_size)
        mask = mask.astype(np.bool)
    return mask


def fill_holes_in_mask(mask, volume):
    """:rtype: np.ndarray"""
    holes_mask = (mask == 0).astype(np.uint8)
    component_mask = sitk.GetArrayFromImage(
        sitk.RelabelComponent(sitk.ConnectedComponent(sitk.GetImageFromArray(holes_mask))))
    border_set = set()
    for dim_num in range(component_mask.ndim):
        border_set.update(np.unique(np.take(component_mask, [0, -1], axis=dim_num)))
    if 0 in border_set:
        border_set.remove(0)
    components_num = component_mask.max()
    assert component_mask.dtype.type(components_num) < component_mask.dtype.type(components_num+1)
    for num in border_set:
        component_mask[component_mask == num] = components_num + 1
    if volume > 0:
        sizes = np.bincount(component_mask.flat)
        for i, v in enumerate(sizes[1:], 1):
            if v < volume and i < components_num + 1:
                component_mask[component_mask == i] = 0
    else:
        component_mask[component_mask <= components_num] = 0
    return component_mask == 0


def fill_2d_holes_in_mask(mask: np.ndarray, volume: int) -> np.ndarray:
    """
    :param mask: mask to fill holes
    :param volume: minimum volume
    """
    mask = np.copy(mask)
    if mask.ndim == 2:
        return fill_holes_in_mask(mask, volume)
    for i in range(mask.shape[0]):
        mask[i] = fill_holes_in_mask(mask[i], volume)
    return mask
