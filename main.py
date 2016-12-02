if __name__ == '__main__':
    from gui import MainWindow
    from PyQt4.QtGui import QApplication
    import sys
    import logging
    logging.basicConfig(level=logging.DEBUG)
    myApp = QApplication(sys.argv)
    wind = MainWindow("PartSeg")
    wind.show()
    myApp.exec_()
    sys.exit()
