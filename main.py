if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    from gui import MainWindow, use_qt5
    if use_qt5:
        from PyQt5.QtWidgets import QApplication
    else:
        from PyQt4.QtGui import QApplication
    import sys
    myApp = QApplication(sys.argv)
    wind = MainWindow("PartSeg", sys.argv)
    wind.show()
    myApp.exec_()
    sys.exit()
