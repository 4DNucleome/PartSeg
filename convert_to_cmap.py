from __future__ import print_function


def spacing(s):
    try:
        x, y, z = map(int, s.split(','))
        return x, y, z
    except:
        raise argparse.ArgumentTypeError("Spacing must be x,y,z")

if __name__ == '__main__':
    import argparse
    import glob
    import os
    import backend
    import numpy as np
    parser = argparse.ArgumentParser("Convert project to chimera cmap")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed")
    parser.add_argument("dest_folder", type=str, nargs=1, help="Destination folder")
    parser.add_argument("-s", "--spacing", dest="spacing", default=None, type=spacing)
    args = parser.parse_args()
    if os.path.isdir(args.source_folder[0]):
        files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.gz"))
    else:
        files_to_proceed = args.source_folder

    settings = backend.Settings("settings.json")
    segment = backend.Segment(settings)

    def canvas_update(image):
        segment.draw_canvas = np.zeros(image.shape, dtype=np.uint8)
        segment.set_image(image)

    settings.add_image_callback(canvas_update)
    num = len(files_to_proceed)
    for i, file_path in enumerate(files_to_proceed):
        file_name = os.path.basename(file_path)
        print("file: {}; {} from {}".format(file_name, i, num))
        backend.load_project(file_path, settings, segment)
        segment.threshold_updated()
        file_name = os.path.splitext(file_name)[0]
        file_name += ".cmap"
        if args.spacing is not None:
            settings.spacing = args.spacing
        backend.save_to_cmap(os.path.join(args.dest_folder[0], file_name), settings, segment)
