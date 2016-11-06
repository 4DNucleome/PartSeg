from __future__ import print_function


def spacing(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Spacing must be x,y,z")

if __name__ == '__main__':
    import tarfile
    import argparse
    import os
    import json
    import tempfile

    parser = argparse.ArgumentParser("Convert project to chimera cmap")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed or one file")
    parser.add_argument("dest_folder", type=str, nargs=1, help="Destination folder")
    parser.add_argument("-s", "--spacing", dest="spacing", default=None, type=spacing)
    args = parser.parse_args()

    if os.path.isdir(args.source_folder[0]):
        files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.gz"))
    else:
        files_to_proceed = args.source_folder
    num = len(files_to_proceed)
    for i, file_path in enumerate(files_to_proceed):
        file_name = os.path.basename(file_path)
        print("file: {}; {} from {}".format(file_name, i+1, num))
        folder_path = tempfile.mkdtemp()
        tar = tarfile.open(file_path, 'r:bz2')
        members = tar.getnames()
        important_data = json.load(tar.extractfile("data.json"))
        for name in members:
            if name == "data.json":
                continue
            tar.extract(name, os.path.join(folder_path, name))
        tar.close()
        if args.spacing is not None:
            important_data["spacing"] = args.spacing
        with open(os.path.join(folder_path, "data.json"), 'w') as ff:
            json.dump(important_data, ff)
        tar = tarfile.open(os.path.join(args.dest_folder[0], file_name), 'w:bz2')
        for name in os.listdir(folder_path):
            tar.add(os.path.join(folder_path, name), name)
        tar.close()


