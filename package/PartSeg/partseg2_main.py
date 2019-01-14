import sys
__author__ = "Grzegorz Bokota"
from project_utils_qt.except_hook import my_excepthook
import multiprocessing
multiprocessing.freeze_support()

sys.excepthook = my_excepthook


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    from PyQt5.QtWidgets import QApplication
    myApp = QApplication(sys.argv)
    from partseg_utils.base_argparser import CustomParser
    args = CustomParser("PartSeg").parse_args()
    print(args)
    from segmentation_analysis.main_window import MainWindow
    wind = MainWindow("PartSeg")
    wind.show()
    myApp.exec_()
    del wind
    del myApp
    sys.exit()

if __name__ == '__main__':
     main()