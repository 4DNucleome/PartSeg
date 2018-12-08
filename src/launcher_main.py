import argparse
import sys
import logging

from PyQt5.QtWidgets import QApplication

from partseg_utils.base_argparser import CustomParser
from project_utils_qt.except_hook import my_excepthook

import multiprocessing
multiprocessing.freeze_support()

sys.excepthook = my_excepthook
print("buka")
def main():
    parser = CustomParser("PartSeg")
    parser.add_argument("--multiprocessing-fork", dest="mf", action="store_true",
                        help=argparse.SUPPRESS) # Windows bugfix
    sp = parser.add_subparsers()
    sp_a = sp.add_parser("analysis", help="Starts GUI for analysis")
    sp_s = sp.add_parser("segmentation", help="Starts GUI for segmentation")
    parser.set_defaults(gui="launcher")
    sp_a.set_defaults(gui="analysis")
    sp_s.set_defaults(gui="segmentation")
    argv = [x for x in sys.argv[1:] if not (x.startswith("parent") or x.startswith("pipe"))]
    args = parser.parse_args(argv)
    print(args)

    logging.basicConfig(level=logging.INFO)
    myApp = QApplication(sys.argv)
    if args.gui == "analysis" or args.mf:
        from partseg2.main_window import MainWindow
        title = "PartSeg"
    elif args.gui == "segmentation":
        from stackseg.stack_gui_main import MainWindow
        title = "StackSeg"
    else:
        from launcher.main_window import MainWindow
        title = "Launcher"
    wind = MainWindow(title)
    wind.show()
    myApp.exec_()
    sys.exit()

if __name__ == '__main__':
    main()