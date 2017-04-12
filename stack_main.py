
__author__ = "Grzegorz Bokota"


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    from stack_gui_main import MainWindow
    from qt_import import QApplication
    import sys
    myApp = QApplication(sys.argv)
    wind = MainWindow("StackSeg")
    wind.show()
    myApp.exec_()
    sys.exit()
