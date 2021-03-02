import argparse
import logging
import multiprocessing
import os
import platform
import sys
from functools import partial

from qtpy.QtCore import Qt
from qtpy.QtGui import QFontDatabase

from PartSeg import ANALYSIS_NAME, APP_NAME, MASK_NAME
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
    import napari
    from packaging.version import parse

    if parse(napari.__version__) < parse("0.4.5"):
        from napari._qt.widgets.qt_console import QtConsole
    else:
        from napari_console.qt_console import QtConsole

    from PartSeg import plugins
    from PartSeg._launcher.main_window import MainWindow
    from PartSeg._roi_analysis.main_window import MainWindow as AnalysisMain
    from PartSeg._roi_mask.main_window import MainWindow as MaskMain
    from PartSeg.common_backend.base_argparser import _setup_sentry

    _setup_sentry()
    plugins.register()
    w1 = AnalysisMain("test")
    w2 = MaskMain("test")
    w3 = MainWindow("test")
    if parse(napari.__version__) < parse("0.4.5"):
        console = QtConsole()
    else:
        console = QtConsole(napari.Viewer())
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
    sp_s.set_defaults(gui="roi_mask")
    sp_a.add_argument("image", nargs="?", help="image to read on begin", default="")
    sp_a.add_argument("mask", nargs="?", help="mask to read on begin", default=None)
    sp_a.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    sp_s.add_argument("image", nargs="?", help="image to read on begin", default="")
    argv = [x for x in sys.argv[1:] if not (x.startswith("parent") or x.startswith("pipe"))]
    args = parser.parse_args(argv)
    # print(args)

    logging.basicConfig(level=logging.INFO)
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    CustomApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    my_app = CustomApplication(sys.argv, name="PartSeg", icon=os.path.join(icons_dir, "icon.png"))
    my_app.check_release()
    try:
        from napari.qt import get_app

        get_app()
    except ImportError:
        pass
    QFontDatabase.addApplicationFont(os.path.join(font_dir, "Symbola.ttf"))
    if args.gui == "roi_analysis" or args.mf:
        from PartSeg import plugins

        plugins.register()
        from PartSeg._roi_analysis.main_window import MainWindow

        title = f"{APP_NAME} {ANALYSIS_NAME}"
        if args.image:
            image = TiffImageReader.read_image(args.image, args.mask)
            MainWindow = partial(MainWindow, initial_image=image)
        wind = MainWindow(title=title)
        if args.batch:
            wind.main_menu.batch_window()
    elif args.gui == "roi_mask":
        from PartSeg import plugins

        plugins.register()
        from PartSeg._roi_mask.main_window import MainWindow

        title = f"{APP_NAME} {MASK_NAME}"
        if args.image:
            image = TiffImageReader.read_image(args.image)
            MainWindow = partial(MainWindow, initial_image=image)
        wind = MainWindow(title=title)
    else:
        from PartSeg._launcher.main_window import MainWindow

        title = f"{APP_NAME} Launcher"
        wind = MainWindow(title=title)

    try:
        from napari._qt.qthreading import wait_for_workers_to_quit
    except ImportError:
        from napari._qt.threading import wait_for_workers_to_quit
    my_app.aboutToQuit.connect(wait_for_workers_to_quit)
    wind.show()
    rc = my_app.exec_()
    del wind
    del my_app
    sys.exit(rc)


if __name__ == "__main__":
    main()
