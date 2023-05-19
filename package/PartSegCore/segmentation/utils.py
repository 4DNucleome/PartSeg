import numpy as np
import SimpleITK as sitk


def close_small_holes(image, max_hole_size):
    if image.dtype == bool:
        image = image.astype(np.uint8)
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[0] == 1):
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(image)), True)
        return sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    for layer in image:
        rev_conn = sitk.ConnectedComponent(sitk.BinaryNot(sitk.GetImageFromArray(layer)), True)
        layer[...] = sitk.GetArrayFromImage(sitk.BinaryNot(sitk.RelabelComponent(rev_conn, max_hole_size)))
    return image
