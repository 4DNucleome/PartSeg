import dataclasses
import logging
import os
from argparse import ArgumentParser
from glob import glob
from math import ceil

import numpy as np

from PartSegCore.mask.io_functions import LoadStackImage, MaskProjectTuple, SaveROI, SaveROIOptions
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image, ImageWriter


def generate_mask(project_tuple: MaskProjectTuple, size: int, save_path: str):
    image = project_tuple.image
    count_components = ceil(image.shape[-1] / size) * ceil(image.shape[-2] / size)
    logging.info("Generate mask with %s components", count_components)
    if count_components > 254:
        dtype = np.uint16
    else:
        dtype = np.uint8
    mask = np.zeros(image.shape, dtype=dtype)
    cnt = 1
    for i in range(ceil(image.shape[-1] / size)):
        for j in range(ceil(image.shape[-2] / size)):
            mask[..., j * size : (j + 1) * size, i * size : (i + 1) * size] = cnt
            cnt += 1
    project_tuple = dataclasses.replace(project_tuple, roi_info=ROIInfo(mask))
    logging.info("Save mask to %s", save_path)
    SaveROI.save(save_path, project_tuple, SaveROIOptions(relative_path=True, mask_data=True))


def cut_image(image: Image, size: int, save_dir: str):
    num = 1
    for i in range(ceil(image.shape[-1] / size)):
        for j in range(ceil(image.shape[-2] / size)):
            image_cut = image.cut_image(
                [slice(None), slice(None), slice(j * size, (j + 1) * size), slice(i * size, (i + 1) * size)]
            )
            ImageWriter.save(
                image_cut,
                os.path.join(
                    save_dir,
                    os.path.splitext(os.path.basename(image.file_path))[0] + f"_component{num}.tif",
                ),
            )
            num += 1


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
                os.path.splitext(os.path.join(args.save_dir, os.path.basename(file_path)))[0] + "_mask.seg",
            )
        else:
            cut_image(project_tuple.image, args.size, args.save_dir)


if __name__ == "__main__":
    main()
