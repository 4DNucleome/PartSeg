if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    import argparse
    import sys
    import multiprocessing
    import matplotlib
    multiprocessing.freeze_support()
    # import warnings
    # warnings.filterwarnings('error')
    from partseg_utils.global_settings import set_qt4, set_qt5, set_develop
    parser = argparse.ArgumentParser("Program for segment of connected components")
    parser.add_argument("file", nargs="?", help="file to open")
    parser.add_argument("-qt4", "--force-qt4", dest="qt4", const=True, default=False, action="store_const",
                        help="Force to use qt4 as backend, exclude with qt5 option")
    parser.add_argument("-qt5", "--force-qt5", dest="qt5", const=True, default=False, action="store_const",
                        help="Force to use qt5 as backend, exclude with qt4 option")
    parser.add_argument("-d", "--develop", dest="develop", default=False, const=True, action="store_const",
                        help=argparse.SUPPRESS)
    parser.add_argument("--multiprocessing-fork", dest="mf", nargs=1,
                        help=argparse.SUPPRESS) # Windows bugfix
    args = parser.parse_args()
    if args.qt4 and args.qt5:
        parser.print_help()
        parser.exit(-1)
    if args.qt4:
        set_qt4()
        matplotlib.use("Qt4Agg")
    if args.qt5:
        set_qt5()
        matplotlib.use("Qt5Agg")
    set_develop(args.develop)
    from PyQt5.QtWidgets import QApplication
    from partseg_old.main_window import MainWindow
    import sys
    myApp = QApplication(sys.argv)
    wind = MainWindow("PartSeg", args.file)
    wind.show()
    myApp.exec_()
    sys.exit()
