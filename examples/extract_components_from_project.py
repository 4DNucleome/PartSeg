from argparse import ArgumentParser
from glob import glob
from itertools import chain
from pathlib import Path

from PartSegCore.mask.io_functions import LoadROIImage, SaveComponents, SaveComponentsOptions


def cut_components(project_file: Path):
    project = LoadROIImage.load([str(project_file)])
    SaveComponents.save(
        str(project_file.parent / (project_file.stem + "_components")),
        project,
        SaveComponentsOptions(
            frame=0,
            mask_data=True,
        ),
    )


def main():
    parser = ArgumentParser()
    parser.add_argument("project_files", nargs="+", type=str)
    args = parser.parse_args()
    files = list(chain.from_iterable(glob(f) for f in args.project_files))
    for file_path in files:
        print(f"Processing {file_path}")
        cut_components(Path(file_path))


if __name__ == "__main__":
    main()
