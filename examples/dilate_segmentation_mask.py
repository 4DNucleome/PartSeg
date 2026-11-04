"""Example script to dilate segmentation masks in image files."""

import sys
from argparse import ArgumentParser
from glob import glob
from itertools import chain
from pathlib import Path

import numpy as np

from PartSegCore.image_operations import dilate, to_binary_image
from PartSegCore.mask.io_functions import LoadROIImage, MaskProjectTuple, SaveROI, SaveROIOptions
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.watershed import calculate_distances_array, get_neigh
from PartSegCore_compiled_backend.sprawl_utils.find_split import euclidean_sprawl


def convert_mask(file_path: Path, radius: float, suffix: str, only_selected: bool):
    if radius <= 0:
        raise ValueError("Radius must be positive")
    print(f"Converting {file_path} to {suffix} with radius {radius}")

    project = LoadROIImage.load([str(file_path)])

    roi_ = project.roi_info.roi.squeeze()
    selected_components = project.selected_components
    if only_selected and selected_components is not None:
        mask = np.isin(roi_, selected_components)
        roi_ = roi_ * mask

        unique_values = np.unique(roi_)
        mapping = np.zeros(np.max(unique_values) + 1, dtype=roi_.dtype)
        for new_val, old_val in enumerate(unique_values):
            mapping[old_val] = new_val
        roi_ = mapping[roi_]

        selected_components = list(range(1, len(unique_values)))

    bin_roi = to_binary_image(roi_)
    sprawl_area = dilate(bin_roi, [radius, radius], True)
    components_num = np.max(roi_)
    neigh, dist = calculate_distances_array(project.image.spacing, get_neigh(True))
    roi = project.image.fit_array_to_image(
        euclidean_sprawl(
            sprawl_area,
            roi_,
            components_num,
            neigh,
            dist,
        )
    )
    new_file_path = file_path.with_name(file_path.stem + suffix + file_path.suffix)
    print("Saving to ", new_file_path)
    SaveROI.save(
        str(new_file_path),
        MaskProjectTuple(
            file_path=str(new_file_path),
            image=project.image,
            roi_info=ROIInfo(roi),
            spacing=project.spacing,
            frame_thickness=project.frame_thickness,
            selected_components=selected_components,
        ),
        SaveROIOptions(
            relative_path=True,
            mask_data=True,
            frame_thickness=project.frame_thickness,
            spacing=project.spacing,
        ),
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("project_files", nargs="+", type=str)
    parser.add_argument("--dilate", type=int, default=1)
    parser.add_argument("--suffix", type=str, default="_dilated")
    parser.add_argument("--only-selected", action="store_true")

    args = parser.parse_args()

    files = list(chain.from_iterable(glob(x) for x in args.project_files))
    if not files:
        print("No files found")
        return -1

    for file_path in files:
        convert_mask(Path(file_path).absolute(), args.dilate, args.suffix, args.only_selected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
