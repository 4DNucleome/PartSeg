from __future__ import print_function
import logging


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
    import sys
    import backend
    import numpy as np
    parser = argparse.ArgumentParser("Convert project to chimera cmap")
    parser.add_argument("source_folder", type=str, nargs=1, help="Folder with project files to proceed or one file")
    parser.add_argument("dest_folder", type=str, nargs=1, help="Destination folder")
    parser.add_argument("--base_folder", dest="base_folder", type=str, nargs=1, default=None,
                        help="TBD")
    parser.add_argument("-s", "--spacing", dest="spacing", default=None, type=spacing,
                        help="Spacing between pixels saved to cmap")
    parser.add_argument("-g", "--use_2d_gauss", dest="use_2d_gauss", default=backend.GaussUse.no_gauss,
                        const=backend.GaussUse.gauss_2d, action="store_const",
                        help="Apply 2d (x,y) gauss blur data to image before put in cmap")
    parser.add_argument("-g3", "--use_gauss_3d", dest="use_3d_gauss", default=backend.GaussUse.no_gauss,
                        const=backend.GaussUse.gauss_3d, action="store_const",
                        help="Not apply 3d gauss blur data to image before put in cmap")
    parser.add_argument("-ns", "--no_statistics", dest="no_statistics", default=False, const=True,
                        action="store_const",
                        help="Off saving statistics in 'Chimera/image1/Statistics' group")
    parser.add_argument("-nc", "--no_center_data", dest="no_center_data", default=False, const=True,
                        action="store_const",
                        help="Off centering and rotating volumetric data")
    args = parser.parse_args()
    files_to_proceed = glob.glob(os.path.join(args.source_folder[0], "*.gz"))
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

    settings = backend.Settings("settings.json")
    segment = backend.Segment(settings)

    def canvas_update(image):
        segment.draw_canvas = np.zeros(image.shape, dtype=np.uint8)
        segment.set_image(image)

    settings.add_image_callback(canvas_update)
    num = len(files_to_proceed)
    for i, file_path in enumerate(files_to_proceed):
        if base_folder is not None:
            rel_path = os.path.dirname(os.path.relpath(file_path, base_folder))
        else:
            rel_path = ""
        file_name = os.path.basename(file_path)
        print("file: {}; {} from {}".format(file_name, i+1, num))
        backend.load_project(file_path, settings, segment)
        segment.threshold_updated()
        file_name = os.path.splitext(file_name)[0]
        file_name += ".cmap"
        if args.spacing is not None:
            settings.spacing = args.spacing
        if not os.path.isdir(os.path.join(args.dest_folder[0], rel_path)):
            os.makedirs(os.path.join(args.dest_folder[0], rel_path))
        gauss_type = max(args.use_2d_gauss, args.use_3d_gaus)
        backend.save_to_cmap(os.path.join(args.dest_folder[0], rel_path, file_name), settings, segment,
                             gaus_type=gauss_type, with_statistics=not args.no_statistics,
                             centered_data=not args.no_center_data)
