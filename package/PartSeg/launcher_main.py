import argparse
import sys
import logging
from functools import partial

from PartSeg.tiff_image import ImageReader
from .custom_application import CustomApplication
from PartSeg.project_utils_qt.base_argparser import CustomParser
import multiprocessing

multiprocessing.freeze_support()


def main():
    parser = CustomParser("PartSeg")
    parser.add_argument("--multiprocessing-fork", dest="mf", action="store_true",
                        help=argparse.SUPPRESS)  # Windows bug fix
    sp = parser.add_subparsers()
    sp_a = sp.add_parser("segmentation_analysis", help="Starts GUI for segmentation analysis")
    sp_s = sp.add_parser("segmentation", help="Starts GUI for segmentation")
    parser.set_defaults(gui="launcher")
    sp_a.set_defaults(gui="segmentation_analysis")
    sp_s.set_defaults(gui="segmentation")
    sp_a.add_argument("image", nargs="?", help="image to read on begin", default="")
    sp_a.add_argument("mask", nargs="?", help="mask to read on begin", default=None)
    sp_s.add_argument("image", nargs="?", help="image to read on begin", default="")
    argv = [x for x in sys.argv[1:] if not (x.startswith("parent") or x.startswith("pipe"))]
    args = parser.parse_args(argv)
    # print(args)

    logging.basicConfig(level=logging.INFO)
    my_app = CustomApplication(sys.argv)
    my_app.check_release()
    if args.gui == "segmentation_analysis" or args.mf:
        from . import plugins
        plugins.register()
        from .segmentation_analysis.main_window import MainWindow
        title = "PartSeg Segmentation Analysis"
        if args.image:
            image = ImageReader.read_image(args.image, args.mask)
            MainWindow = partial(MainWindow, initial_image=image)
    elif args.gui == "segmentation":
        from . import plugins
        plugins.register()
        from .segmentation_mask.stack_gui_main import MainWindow
        title = "PartSeg Mask Segmentation"
        if args.image:
            image = ImageReader.read_image(args.image)
            MainWindow = partial(MainWindow, initial_image=image)
    else:
        from .launcher.main_window import MainWindow
        title = "PartSeg Launcher"
    wind = MainWindow(title=title)
    wind.show()
    my_app.exec_()
    del wind
    del my_app
    sys.exit()


if __name__ == '__main__':
    main()
