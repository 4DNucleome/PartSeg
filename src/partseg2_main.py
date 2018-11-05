import sys
__author__ = "Grzegorz Bokota"
from project_utils.except_hook import my_excepthook

sys.excepthook = my_excepthook


def main():
    import logging
    logging.basicConfig(level=logging.INFO)
    from PyQt5.QtWidgets import QApplication
    myApp = QApplication(sys.argv)
    from partseg2.main_window import MainWindow
    wind = MainWindow("PartSeg")
    wind.show()
    myApp.exec_()
    del wind
    del myApp
    sys.exit()

if __name__ == '__main__':
     main()