import typing
from io import BytesIO
from pathlib import Path

import numpy as np
from tifffile import imwrite

from .image import Image, minimal_dtype


class ImageWriter:
    """class for saving TIFF images"""

    @classmethod
    def save(cls, image: Image, save_path: typing.Union[str, BytesIO, Path]):
        """
        Save image as tiff to path or buffer

        :param image: image for save
        :param save_path: save location
        """
        # print(f"[save] {save_path}")
        data = image.get_image_for_save()
        spacing = image.get_um_spacing()
        metadata = {"mode": "color", "unit": "\\u00B5m"}
        if len(spacing) == 3:
            metadata.update({"spacing": spacing[0]})
        if image.channel_names is not None:
            metadata["Labels"] = image.channel_names * image.layers
        coloring = image.get_imagej_colors()
        if coloring is not None:
            metadata["LUTs"] = coloring
        ranges = image.get_ranges()
        ranges = np.array(ranges).reshape(len(ranges) * 2)
        # print(ranges)
        metadata["Ranges"] = ranges

        resolution = [1 / x for x in spacing[-2:]]
        cls._save(data, save_path, resolution, metadata)

    @classmethod
    def save_mask(cls, image: Image, save_path: typing.Union[str, Path]):
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
        metadata = {"mode": "color", "unit": "\\u00B5m"}
        spacing = image.get_um_spacing()
        if len(spacing) == 3:
            metadata.update({"spacing": spacing[0]})
        resolution = [1 / x for x in spacing[-2:]]
        cls._save(mask, save_path, resolution, metadata)

    @staticmethod
    def _save(data: np.ndarray, save_path, resolution=None, metadata=None):
        # TODO change to ome TIFF
        if data.dtype in [np.uint8, np.uint16, np.float32]:
            imwrite(
                save_path,
                data,
                imagej=True,
                software="PartSeg",
                metadata=metadata,
                resolution=resolution,
            )  # , compress=6,
        else:
            raise ValueError(f"Data type {data.dtype} not supported by imagej tiff")
            # imagej=True, software="PartSeg")
