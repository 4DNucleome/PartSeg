if __name__ == '__main__':
    from gui import MainWindow
    from PySide.QtGui import QApplication
    import sys
    myApp = QApplication(sys.argv)
    wind = MainWindow("Part segment")
    wind.show()
    myApp.exec_()
    sys.exit()
