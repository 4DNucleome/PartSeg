import argparse
import logging
import multiprocessing
import os
import platform
import sys
from contextlib import suppress
from functools import partial

multiprocessing.freeze_support()


# noinspection PyUnresolvedReferences,PyUnusedLocal
def _test_imports():  # pragma: no cover
    print("start_test_import")
    from qtpy.QtWidgets import QApplication

    app = QApplication([])
    import freetype
    import napari
    from napari_console.qt_console import QtConsole

    from PartSeg import plugins
    from PartSeg._launcher.main_window import MainWindow
    from PartSeg._roi_analysis.main_window import MainWindow as AnalysisMain
    from PartSeg._roi_mask.main_window import MainWindow as MaskMain
    from PartSeg.common_backend.base_argparser import _setup_sentry
    from PartSeg.common_gui.label_create import LabelChoose
    from PartSeg.plugins import napari_widgets
    from PartSegCore import napari_plugins

    if "BorderSmooth" not in dir(napari_widgets):
        raise ImportError("napari_widgets not loaded")

    if "load_image" not in dir(napari_plugins):
        raise ImportError("napari_plugins not loaded")

    with suppress(ImportError):
        from napari.qt import get_app

        get_app()

    _setup_sentry()
    freetype.get_handle()
    plugins.register()
    w1 = AnalysisMain("test")
    w2 = MaskMain("test")
    w3 = MainWindow("test")
    v = napari.Viewer()
    console = QtConsole(v)
    label = LabelChoose(w1.settings)
    label.refresh()
    v.close()
    del label  # skipcq: PTC-W0043
    del w1  # skipcq: PTC-W0043
    del w2  # skipcq: PTC-W0043
    del w3  # skipcq: PTC-W0043
    del v  # skipcq: PTC-W0043
    del app  # skipcq: PTC-W0043
    del console  # skipcq: PTC-W0043
    print("end_test_import")


def create_parser():
    from PartSeg.common_backend.base_argparser import CustomParser

    parser = CustomParser("PartSeg")
    parser.add_argument(
        "--multiprocessing-fork", dest="mf", action="store_true", help=argparse.SUPPRESS
    )  # Windows bug fix
    sp = parser.add_subparsers()
    sp_a = sp.add_parser("roi_analysis", aliases=["roi"], help="Starts GUI for segmentation analysis")
    sp_s = sp.add_parser("mask_segmentation", aliases=["mask"], help="Starts GUI for segmentation")
    parser.set_defaults(gui="launcher")
    sp_a.set_defaults(gui="roi_analysis")
    sp_s.set_defaults(gui="roi_mask")
    sp_a.add_argument("image", nargs="?", help="image to read on begin", default="")
    sp_a.add_argument("mask", nargs="?", help="mask to read on begin", default=None)
    sp_a.add_argument("--batch", action="store_true", help=argparse.SUPPRESS)
    sp_s.add_argument("image", nargs="?", help="image to read on begin", default="")

    return parser


def main():  # pragma: no cover  # noqa: PLR0915
    from importlib.metadata import version

    from packaging.version import parse as parse_version

    napari_version = parse_version(version("napari"))
    pydantic_version = parse_version(version("pydantic"))
    if napari_version < parse_version("0.4.19") and pydantic_version >= parse_version("2"):
        print("napari version is too low, please update to version 0.4.19 or higher or downgrade pydantic to version 1")
        sys.exit(1)

    if len(sys.argv) > 1 and sys.argv[1] == "_test":
        _test_imports()
        return

    parser = create_parser()

    argv = [x for x in sys.argv[1:] if not x.startswith(("parent", "pipe"))]
    args = parser.parse_args(argv)

    try:
        from qtpy import QT5
    except ImportError:  # pragma: no cover
        QT5 = True
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QIcon
    from qtpy.QtWidgets import QApplication

    from PartSeg import state_store
    from PartSeg._launcher.check_survey import CheckSurveyThread
    from PartSeg._launcher.check_version import CheckVersionThread
    from PartSeg.common_backend import napari_get_settings
    from PartSegData import icons_dir

    logging.basicConfig(level=logging.INFO)
    if platform.system() == "Darwin":
        multiprocessing.set_start_method("spawn")

    if QT5:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    my_app = QApplication(sys.argv)
    my_app.setApplicationName("PartSeg")
    my_app.setWindowIcon(QIcon(os.path.join(icons_dir, "icon.png")))

    napari_get_settings(os.path.join(os.path.dirname(state_store.save_folder), "napari"))
    with suppress(ImportError):
        from napari.qt import get_app

        get_app()

    wind = select_window(args)

    try:
        from napari._qt.qthreading import wait_for_workers_to_quit
    except ImportError:
        from napari._qt.threading import wait_for_workers_to_quit
    my_app.aboutToQuit.connect(wait_for_workers_to_quit)
    check_version = CheckVersionThread()
    check_version.start()
    check_survey = CheckSurveyThread()
    check_survey.start()
    wind.show()
    rc = my_app.exec_()
    del wind  # skipcq: PTC-W0043
    del my_app  # skipcq: PTC-W0043
    sys.exit(rc)


def select_window(args):
    from PartSeg import ANALYSIS_NAME, APP_NAME, MASK_NAME
    from PartSegImage import TiffImageReader

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
        return wind
    if args.gui == "roi_mask":
        from PartSeg import plugins

        plugins.register()
        from PartSeg._roi_mask.main_window import MainWindow

        title = f"{APP_NAME} {MASK_NAME}"
        if args.image:
            image = TiffImageReader.read_image(args.image)
            MainWindow = partial(MainWindow, initial_image=image)
        return MainWindow(title=title)

    from PartSeg._launcher.main_window import MainWindow

    title = f"{APP_NAME} Launcher"
    return MainWindow(title=title)


if __name__ == "__main__":
    main()
