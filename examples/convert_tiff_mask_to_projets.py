import dataclasses
import os
import sys
from glob import glob
from os import path

from tqdm import tqdm

from PartSegCore.mask.io_functions import LoadROIFromTIFF, LoadStackImage, SaveROI, SaveROIOptions


def main():
    file_list = glob("2022-10-22/*/*.obsep")
    for image_path in tqdm(file_list):
        tiff_roi_path = path.join(path.dirname(image_path), "segmentation.tif")
        roi_path = path.join(path.dirname(image_path), "segmentation.seg")
        if os.path.exists(roi_path):
            continue
        if not path.exists(tiff_roi_path):
            print(f"Mask {tiff_roi_path} not found", file=sys.stderr)
            continue
        try:
            project_tuple = LoadStackImage.load([image_path])
            mask_tuple = LoadROIFromTIFF.load([tiff_roi_path])

            project_tuple = dataclasses.replace(project_tuple, roi_info=mask_tuple.roi_info)
            SaveROI.save(roi_path, project_tuple, SaveROIOptions(relative_path=True, mask_data=True))
        except Exception as e:
            print(f"Error in {image_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
