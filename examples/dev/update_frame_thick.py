import argparse
import io
import json
import shutil
import sys
import tarfile
from glob import glob
from pathlib import Path

from PartSegCore.io_utils import get_tarinfo, load_metadata_base
from PartSegCore.json_hooks import PartSegEncoder

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, total):  # noqa: ARG001
        return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="input directory")
    parser.add_argument("frame_thickness", help="frame thickness", type=int)
    args = parser.parse_args()
    files = list(glob(f"{args.input}/*.seg"))
    if not len(files):
        raise print(f"No files in {args.input}/*.seg", file=sys.stderr)

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
        except:
            shutil.move(new_path, file_path)
            raise
        shutil.move(new_path, new_path.with_suffix(".seg_old"))


if __name__ == "__main__":
    main()
