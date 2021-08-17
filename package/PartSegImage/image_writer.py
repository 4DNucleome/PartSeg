import typing
from io import BytesIO
from itertools import product
from pathlib import Path

import numpy as np
from tifffile import imwrite

from .image import Image, minimal_dtype


class ImageWriter:
    """class for saving TIFF images"""

    @classmethod
    def prepare_metadata(cls, image: Image, channels: int):
        spacing = image.get_um_spacing()
        shift = image.get_um_shift()
        metadata = {
            "Pixels": {
                "PhysicalSizeZ": spacing[0] if len(spacing) == 3 else 1,
                "PhysicalSizeY": spacing[-2],
                "PhysicalSizeX": spacing[-1],
                "PhysicalSizeZUnit": "µm",
                "PhysicalSizeYUnit": "µm",
                "PhysicalSizeXUnit": "µm",
            },
            "Plane": [],
            "Creator": "PartSeg",
        }
        if image.name:
            metadata["Name"] = image.name
        for t, z, c in product(range(image.times), range(image.layers), range(channels)):
            metadata["Plane"].append(
                {
                    "TheT": t,
                    "TheZ": z,
                    "TheC": c,
                    "PositionZ": shift[0] if len(shift) == 3 else 0,
                    "PositionY": shift[-2],
                    "PositionX": shift[-1],
                    "PositionZUnit": "µm",
                    "PositionYUnit": "µm",
                    "PositionXUnit": "µm",
                }
            )

        return metadata

    @classmethod
    def save(cls, image: Image, save_path: typing.Union[str, BytesIO, Path], compression="ADOBE_DEFLATE"):
        """
        Save image as tiff to path or buffer

        :param image: image for save
        :param save_path: save location
        """
        # print(f"[save] {save_path}")
        data = image.get_image_for_save()

        metadata = cls.prepare_metadata(image, image.channels)

        metadata["Channel"] = {
            "Name": image.channel_names,
            "axes": "TZYXC",
        }
        cls._save(data, save_path, metadata, compression)

    @classmethod
    def save_mask(cls, image: Image, save_path: typing.Union[str, Path], compression="ADOBE_DEFLATE"):
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
        metadata = cls.prepare_metadata(image, 1)
        metadata["Channel"] = {
            "Name": "Mask",
            "axes": "TZYX",
        }
        cls._save(mask, save_path, metadata, compression)

    @staticmethod
    def _save(data: np.ndarray, save_path, metadata=None, compression="ADOBE_DEFLATE"):
        # TODO change to ome TIFF
        imwrite(
            save_path,
            data,
            ome=True,
            software="PartSeg",
            metadata=metadata,
            compression=compression,
        )  # , compress=6,
