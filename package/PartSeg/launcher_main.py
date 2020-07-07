import argparse
import logging
import multiprocessing
import os
import sys
from functools import partial

from napari._qt.threading import wait_for_workers_to_quit
from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase

from PartSeg import APP_NAME, MASK_NAME, SEGMENTATION_NAME
from PartSeg.common_backend.base_argparser import CustomParser
from PartSeg.custom_application import CustomApplication
from PartSegData import font_dir, icons_dir
from PartSegImage import TiffImageReader

multiprocessing.freeze_support()


# noinspection PyUnresolvedReferences,PyUnusedLocal
def _test_imports():
    print("start_test_import")
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    from PartSeg.segmentation_analysis.main_window import MainWindow as AnalysisMain
    from PartSeg.segmentation_mask.main_window import MainWindow as MaskMain
    from PartSeg.launcher.main_window import MainWindow
    from PartSeg import plugins
    from napari._qt.qt_console import QtConsole

    plugins.register()
    w1 = AnalysisMain("test")
    w2 = MaskMain("test")
    w3 = MainWindow("test")
    console = QtConsole()
    if QFontDatabase.addApplicationFont(os.path.join(font_dir, "Symbola.ttf")) == -1:
        raise ValueError("Error with loading Symbola font")
    del w1
    del w2
    del w3
    del app
    del console
    print("end_test_import")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_test":
        _test_imports()
        return
    parser = CustomParser("PartSeg")
    parser.add_argument(
        "--multiprocessing-fork", dest="mf", action="store_true", help=argparse.SUPPRESS
    )  # Windows bug fix
    sp = parser.add_subparsers()
    sp_a = sp.add_parser("roi_analysis", help="Starts GUI for segmentation analysis")
    sp_s = sp.add_parser("mask_segmentation", help="Starts GUI for segmentation")
    parser.set_defaults(gui="launcher")
    sp_a.set_defaults(gui="roi_analysis")
    sp_s.set_defaults(gui="mask_segmentation")
    sp_a.add_argument("image", nargs="?", help="image to read on begin", default="")
    sp_a.add_argument("mask", nargs="?", help="mask to read on begin", default=None)
    sp_a.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    sp_s.add_argument("image", nargs="?", help="image to read on begin", default="")
    argv = [x for x in sys.argv[1:] if not (x.startswith("parent") or x.startswith("pipe"))]
    args = parser.parse_args(argv)
    # print(args)

    logging.basicConfig(level=logging.INFO)
    CustomApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    my_app = CustomApplication(sys.argv, name="PartSeg", icon=os.path.join(icons_dir, "icon.png"))
    my_app.check_release()
    my_app.aboutToQuit.connect(wait_for_workers_to_quit)
    QFontDatabase.addApplicationFont(os.path.join(font_dir, "Symbola.ttf"))
    if args.gui == "roi_analysis" or args.mf:
        from PartSeg import plugins

        plugins.register()
        from PartSeg.segmentation_analysis.main_window import MainWindow

        title = f"{APP_NAME} {SEGMENTATION_NAME}"
        if args.image:
            image = TiffImageReader.read_image(args.image, args.mask)
            MainWindow = partial(MainWindow, initial_image=image)
        wind = MainWindow(title=title)
        if args.batch:
            wind.main_menu.batch_window()
    elif args.gui == "mask_segmentation":
        from PartSeg import plugins

        plugins.register()
        from PartSeg.segmentation_mask.main_window import MainWindow

        title = f"{APP_NAME} {MASK_NAME}"
        if args.image:
            image = TiffImageReader.read_image(args.image)
            MainWindow = partial(MainWindow, initial_image=image)
        wind = MainWindow(title=title)
    else:
        from PartSeg.launcher.main_window import MainWindow

        title = f"{APP_NAME} Launcher"
        wind = MainWindow(title=title)

    wind.show()
    rc = my_app.exec_()
    del wind
    del my_app
    sys.exit(rc)


if __name__ == "__main__":
    main()
