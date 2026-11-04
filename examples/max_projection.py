"""
For collections of tiff files save a max projection of each file.
"""

from argparse import ArgumentParser
from glob import glob
from itertools import chain
from pathlib import Path

from PartSegImage import Image, ImageWriter, TiffImageReader


def max_projection(file_path: Path, suffix: str = "_max", with_mask: bool = False):
    if with_mask:
        mask_path = str(file_path.parent / (file_path.stem + "_mask" + file_path.suffix))
    else:
        mask_path = None
    image = TiffImageReader.read_image(str(file_path), mask_path)
    if "Z" not in image.axis_order:
        raise ValueError(f"Image {file_path} does not have Z axis")
    max_proj = image.get_data().max(axis=image.axis_order.index("Z"))
    if with_mask:
        mask_projection = image.mask.max(axis=image.array_axis_order.index("Z"))
    else:
        mask_projection = None
    image2 = Image(
        max_proj, spacing=image.spacing[1:], axes_order=image.axis_order.replace("Z", ""), mask=mask_projection
    )
    ImageWriter.save(image2, str(file_path.with_name(file_path.stem + suffix + file_path.suffix)))
    if with_mask:
        ImageWriter.save_mask(image2, str(file_path.with_name(file_path.stem + suffix + "_mask" + file_path.suffix)))


def main():
    parser = ArgumentParser()
    parser.add_argument("image_files", nargs="+", type=str)
    parser.add_argument("--suffix", type=str, default="_max")
    parser.add_argument("--with-mask", action="store_true")
    args = parser.parse_args()
    files = list(chain.from_iterable(glob(f) for f in args.image_files))
    for file_path in files:
        if args.with_mask and Path(file_path).stem.endswith("_mask"):
            continue
        print(f"Processing {file_path}")
        max_projection(Path(file_path), args.suffix, args.with_mask)


if __name__ == "__main__":
    main()
