import argparse
import sys
import logging

from PyQt5.QtWidgets import QApplication

from project_utils.except_hook import my_excepthook

sys.excepthook = my_excepthook

def main():
    parser = argparse.ArgumentParser("PartSeg")
    sp = parser.add_subparsers()
    sp_a = sp.add_parser("analysis", help="Starts GUI for analysis")
    sp_s = sp.add_parser("segmentation", help="Starts GUI for segmentation")
    parser.set_defaults(gui="launcher")
    sp_a.set_defaults(gui="analysis")
    sp_s.set_defaults(gui="segmentation")
    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.INFO)
    myApp = QApplication(sys.argv)
    if args.gui == "analysis":
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