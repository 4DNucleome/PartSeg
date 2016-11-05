from __future__ import print_function
if __name__ == '__main__':
    import argparse
    import glob
    import os
    import backend
    import tarfile
    import numpy as np
    import tifffile
    parser = argparse.ArgumentParser("Extract images from project")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed")
    parser.add_argument("dest_folder", type=str, nargs=1, help="Destination folder")
    args = parser.parse_args()
    if os.path.isdir(args.source_folder[0]):
        files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.gz"))
    else:
        files_to_proceed = args.source_folder
    settings = backend.Settings("settings.json")
    segment = backend.Segment(settings)
    for file_path in files_to_proceed:
        file_name = os.path.basename(file_path)
        file_name = os.path.splitext(file_name)[0]
        file_name += ".tiff"
        tar = tarfile.open(file_path, 'r:bz2')
        image = np.load(tar.extractfile("image.npy"))
        tifffile.imsave(os.path.join(args.dest_folder[0], file_name), image)
