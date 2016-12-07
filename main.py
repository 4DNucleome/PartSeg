if __name__ == '__main__':
    from gui import MainWindow
    from PyQt4.QtGui import QApplication
    import sys
    import logging
    logging.basicConfig(level=logging.INFO)
    myApp = QApplication(sys.argv)
    print sys.argv
    wind = MainWindow("PartSeg", sys.argv)
    wind.show()
    myApp.exec_()
    sys.exit()
