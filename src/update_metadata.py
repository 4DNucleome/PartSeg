from __future__ import print_function


def spacing(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Spacing must be x,y,z")


def extension(s):
    ext_list = ["gz", "tgz", "tar.gz", "bz2", "tbz2", "tar.bz2"]
    ext_list.extend(map(lambda x: "."+x, ext_list))
    if s in ext_list:
        if s[0] != ".":
            s = "."+s
        return s
    raise argparse.ArgumentTypeError("Extension must be one of this: {}".format(ext_list))

if __name__ == '__main__':
    import tarfile
    import argparse
    import os
    import json
    import tempfile
    import glob
    import logging
    import sys

    parser = argparse.ArgumentParser("Convert project to chimera cmap")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed or one file")
    parser.add_argument("destination_folder", type=str, nargs=1, help="Destination folder")
    parser.add_argument("-s", "--spacing", dest="spacing", default=None, type=spacing)
    parser.add_argument("-e", "--extension", dest="extension", default=None, type=extension, nargs=1)
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
        if base_folder is not None:
            rel_path = os.path.dirname(os.path.relpath(file_path, base_folder))
        else:
            rel_path = ""
        file_name = os.path.basename(file_path)
        if args.extension is not None:
            file_name = os.path.splitext(file_name)[0] + args.extension[0]
        print("file: {}; {} from {}".format(file_name, i+1, num))
        folder_path = tempfile.mkdtemp()
        if os.path.splitext(file_path)[1] in [".bz2", ".tbz2", ".tar.bz2"]:
            tar = tarfile.open(file_path, 'r:bz2')
        else:
            tar = tarfile.open(file_path, 'r:gz')
        members = tar.getnames()
        important_data = json.load(tar.extractfile("data.json"))
        for name in members:
            if name == "data.json":
                continue
            tar.extract(name, folder_path)
        tar.close()
        if args.spacing is not None:
            important_data["spacing"] = args.spacing
        with open(os.path.join(folder_path, "data.json"), 'w') as ff:
            json.dump(important_data, ff)
        if not os.path.isdir(os.path.join(args.destination_folder[0], rel_path)):
            os.makedirs(os.path.join(args.destination_folder[0], rel_path))

        if os.path.splitext(file_name)[1] in [".bz2", ".tbz2", ".tar.bz2"]:
            tar = tarfile.open(os.path.join(args.destination_folder[0], rel_path, file_name), 'w:bz2')
        else:
            tar = tarfile.open(os.path.join(args.destination_folder[0], rel_path, file_name), 'w:gz')
        for name in os.listdir(folder_path):
            tar.add(os.path.join(folder_path, name), name)
        tar.close()
