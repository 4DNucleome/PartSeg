"""
For collections of tiff files save a max projection of each file.
"""

from argparse import ArgumentParser
from glob import glob
from itertools import chain
from pathlib import Path

from PartSegImage import Image, ImageWriter, TiffImageReader


def max_projection(file_path: Path, suffix: str = "_max"):
    image = TiffImageReader.read_image(str(file_path))
    max_proj = image.get_data().max(axis=image.axis_order.index("Z"))
    image2 = Image(max_proj, image.spacing[1:], axes_order=image.axis_order.replace("Z", ""))
    ImageWriter.save(image2, str(file_path.with_name(file_path.stem + suffix + file_path.suffix)))


def main():
    parser = ArgumentParser()
    parser.add_argument("image_files", nargs="+", type=str)
    parser.add_argument("--suffix", type=str, default="_max")
    args = parser.parse_args()
    files = list(chain.from_iterable(glob(f) for f in args.image_files))
    for file_path in files:
        print(f"Processing {file_path}")
        max_projection(Path(file_path), args.suffix)


if __name__ == "__main__":
    main()
