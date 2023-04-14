"""
This is script to cut larger 2d images into smaller ones. It is useful when you have large image and you want to
be able to process it on weaker computer.
"""

import dataclasses
import logging
import os
from argparse import ArgumentParser
from glob import glob
from itertools import product
from math import ceil

import numpy as np

from PartSegCore.mask.io_functions import LoadStackImage, MaskProjectTuple, SaveROI, SaveROIOptions
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image, ImageWriter

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, total):  # noqa: ARG001
        return x


def generate_mask(project_tuple: MaskProjectTuple, size: int, save_path: str):
    image = project_tuple.image
    if not isinstance(image, Image):  # pragma: no cover
        raise ValueError(f"project_tuple.image must be instance of Image, not {type(image)}")
    count_components = ceil(image.shape[-1] / size) * ceil(image.shape[-2] / size)
    logging.info("Generate mask with %s components", count_components)
    if count_components > 254:
        dtype = np.uint16
    else:
        dtype = np.uint8
    mask = np.zeros(image.shape, dtype=dtype)
    x_step = ceil(image.shape[-1] / size)
    y_step = ceil(image.shape[-2] / size)
    for cnt, (i, j) in enumerate(tqdm(product(range(x_step), range(y_step)), total=x_step * y_step), start=1):
        mask[..., j * size : (j + 1) * size, i * size : (i + 1) * size] = cnt
    project_tuple = dataclasses.replace(project_tuple, roi_info=ROIInfo(mask))
    logging.info("Save mask to %s", save_path)
    SaveROI.save(save_path, project_tuple, SaveROIOptions(relative_path=True, mask_data=True, frame_thickness=0))


def cut_image(image: Image, size: int, save_dir: str):
    x_step = ceil(image.shape[-1] / size)
    y_step = ceil(image.shape[-2] / size)
    for cnt, (i, j) in enumerate(tqdm(product(range(x_step), range(y_step)), total=x_step * y_step), start=1):
        image_cut = image.cut_image(
            [slice(None), slice(None), slice(j * size, (j + 1) * size), slice(i * size, (i + 1) * size)]
        )
        ImageWriter.save(
            image_cut,
            os.path.join(
                save_dir,
                f"{os.path.splitext(os.path.basename(image.file_path))[0]}_component{cnt}.tif",
            ),
        )


def main():
    parser = ArgumentParser()
    parser.add_argument("file", help="File to cut")
    parser.add_argument("save_dir", help="Directory to save")
    parser.add_argument("size", type=int, help="Size of cut")
    parser.add_argument("--mask", help="Generate mask", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    for file_path in glob(args.file):
        logging.info("Processing %s", file_path)
        project_tuple = LoadStackImage.load([file_path])
        if args.mask:
            generate_mask(
                project_tuple,
                args.size,
                f"{os.path.splitext(os.path.join(args.save_dir, os.path.basename(file_path)))[0]}_mask.seg",
            )
        else:
            cut_image(project_tuple.image, args.size, args.save_dir)


if __name__ == "__main__":
    main()
