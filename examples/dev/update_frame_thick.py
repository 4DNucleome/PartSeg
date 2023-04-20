"""
Script to update frame thickness for mask segmentation files.
It could be used to add frame thickness for mask segmentation files created with older versions of PartSeg or to update
it it files contain wrong value.
"""

import argparse
import io
import json
import shutil
import sys
import tarfile
from glob import glob
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x):
        return x


def main():
    parser = argparse.ArgumentParser(
        description="This is script to add or update frame thickness for mask segmentation files. During load"
        " mask segmentation files in PartSeg roi gui additional frame is added to catted objects."
        " The default frame thickness is 2 and was hardcoded in older (before 0.15.0) versions of PartSeg."
        " To avoid expensive recreation of all mask segmentation files this script can be used "
        "to update frame thickness."
    )
    parser.add_argument("input", help="input directory")
    parser.add_argument("frame_thickness", help="frame thickness", type=int)
    args = parser.parse_args()
    process_files(args)


def process_files(args):
    from PartSegCore.io_utils import get_tarinfo, load_metadata_base
    from PartSegCore.json_hooks import PartSegEncoder

    files = list(glob(f"{args.input}/*.seg"))
    if not files:
        print(f"No files in {args.input}/*.seg", file=sys.stderr)
        return

    for file_path in tqdm(files):
        path = Path(file_path)
        new_path = path.with_suffix(".tgz")
        shutil.move(file_path, new_path)
        with tarfile.open(new_path, "r:*") as f:
            metadata = load_metadata_base(f.extractfile("metadata.json").read().decode("utf8"))
            segmentation_buff = io.BytesIO()
            segmentation_tar = f.extractfile("segmentation.tif")
            segmentation_buff.write(segmentation_tar.read())
        try:
            with tarfile.open(file_path, "w:gz") as f:
                metadata["frame_thickness"] = args.frame_thickness
                meta_buff = io.BytesIO(json.dumps(metadata, cls=PartSegEncoder).encode("utf8"))
                f.addfile(get_tarinfo("metadata.json", meta_buff), meta_buff)
                f.addfile(get_tarinfo("segmentation.tif", segmentation_buff), segmentation_buff)
        except:  # noqa: E722 RUF100
            shutil.move(new_path, file_path)
            raise
        shutil.move(new_path, new_path.with_suffix(".seg_old"))


if __name__ == "__main__":
    main()
