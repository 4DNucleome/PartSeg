import tarfile
import argparse
import os
import logging
import sys
import tifffile
import numpy as np
import tempfile
import glob

if sys.version_info.major == 2:
    def extract_numpy_file(name):
        return np.load(tar.extractfile(name))
else:
    folder_path = tempfile.mkdtemp()


    def extract_numpy_file(name):
        tar.extract(name, folder_path)
        return np.load(os.path.join(folder_path, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Extract images from project files")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed or one file")
    parser.add_argument("destination_folder", type=str, nargs=1, help="Destination folder")
    parser.add_argument("--base_folder", dest="base_folder", type=str, nargs=1, default=None,
                        help="TBD")
    args = parser.parse_args()

    files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.gz"))
    if len(files_to_proceed) == 0:
        files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.bz2"))
        if len(files_to_proceed) == 0:
            files_to_proceed = args.source_folder

    if args.base_folder is not None:
        if not os.path.isdir(args.base_folder[0]):
            logging.error("Folder {} does not exists".format(args.base_folder[0]))
            sys.exit(-1)
        else:
            base_folder = args.base_folder[0]
    else:
        base_folder = None

    num = len(files_to_proceed)

    for i, file_path in enumerate(files_to_proceed):
        file_name = os.path.basename(file_path)
        print("file: {}; {} from {}".format(file_name, i+1, num))
        if base_folder is not None:
            rel_path = os.path.dirname(os.path.relpath(file_path, base_folder))
        else:
            rel_path = ""
        try:
            tar = tarfile.open(file_path, 'r:bz2')
        except tarfile.ReadError:
            tar = tarfile.open(file_path, 'r:gz')
        file_name = os.path.splitext(file_name)[0]
        file_name += ".tiff"
        image = extract_numpy_file("image.npy")
        tifffile.imsave(os.path.join(args.destination_folder, rel_path, file_name), image)
