import argparse
import sys
import logging

from .custom_application import CustomApplication

from .partseg_utils.base_argparser import CustomParser
from .project_utils_qt.except_hook import my_excepthook

import multiprocessing

multiprocessing.freeze_support()

sys.excepthook = my_excepthook


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
    argv = [x for x in sys.argv[1:] if not (x.startswith("parent") or x.startswith("pipe"))]
    args = parser.parse_args(argv)
    print(args)

    logging.basicConfig(level=logging.INFO)
    my_app = CustomApplication(sys.argv)
    if args.gui == "segmentation_analysis" or args.mf:
        from . import plugins
        plugins.register()
        from .segmentation_analysis.main_window import MainWindow
        title = "PartSeg Segmentation Analysis"
    elif args.gui == "segmentation":
        from . import plugins
        plugins.register()
        from .segmentation_mask.stack_gui_main import MainWindow
        title = "PartSeg Mask Segmentation"
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
