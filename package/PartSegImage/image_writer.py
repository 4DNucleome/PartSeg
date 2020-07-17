import typing
from io import BytesIO

import numpy as np
from tifffile import imwrite

from .image import Image, minimal_dtype


class ImageWriter:
    """class for saving TIFF images"""

    @classmethod
    def save(cls, image: Image, save_path: typing.Union[str, BytesIO]):
        """
        Save image as tiff to path or buffer

        :param image: image for save
        :param save_path: save location
        """
        # print(f"[save] {save_path}")
        data = image.get_image_for_save()
        imagej_kwargs = {}
        if image.labels is not None:
            imagej_kwargs["Labels"] = image.labels * image.layers
        coloring = image.get_imagej_colors()
        if coloring is not None:
            imagej_kwargs["LUTs"] = coloring
        ranges = image.get_ranges()
        ranges = np.array(ranges).reshape(len(ranges) * 2)
        # print(ranges)
        imagej_kwargs["Ranges"] = ranges
        spacing = image.get_um_spacing()

        metadata = {"mode": "color", "unit": "\\u00B5m"}
        if len(spacing) == 3:
            metadata.update({"spacing": spacing[0]})
        resolution = [1 / x for x in spacing[-2:]]
        cls._save(data, save_path, resolution, metadata, imagej_kwargs)

    @classmethod
    def save_mask(cls, image: Image, save_path: str):
        """
        Save mask connected to image as tiff to path or buffer

        :param image: mast is obtain with :py:meth:`.Image.get_mask_for_save`
        :param save_path: save location
        """
        mask = image.get_mask_for_save()
        if mask is None:
            return
        mask_max = np.max(mask)
        mask = mask.astype(minimal_dtype(mask_max))
        cls._save(mask, save_path)

    @staticmethod
    def _save(data: np.ndarray, save_path, resolution=None, metadata=None, imagej_metadata=None):
        # TODO change to ome TIFF
        if data.dtype in [np.uint8, np.uint16, np.float32]:
            imwrite(
                save_path,
                data,
                imagej=True,
                software="PartSeg",
                metadata=metadata,
                ijmetadata=imagej_metadata,
                resolution=resolution,
            )  # , compress=6,
        else:
            raise ValueError(f"Data type {data.dtype} not supported by imagej tiff")
            # imagej=True, software="PartSeg")
