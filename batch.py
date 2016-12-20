import sys

if sys.version_info.major == 2:
    import pkgutil
    loader = pkgutil.find_loader("PyQt5")
    if loader is not None:
        use_qt5 = True
    else:
        use_qt5 = False
else:
    import importlib
    spam_spec = importlib.util.find_spec("PyQt5")
    if spam_spec is not None:
        use_qt5 = True
    else:
        use_qt5 = False

if use_qt5:
    raise NotImplemented("Pyqt5 support not implemented")
    pass
else:
    from PyQt4.QtCore import Qt
    from PyQt4.QtGui import QWidget, QLineEdit, QPushButton

__author__ = "Grzegorz Bokota"


class BatchWindow(QWidget):
    def __init__(self, parent):
        QWidget.__init__(self, parent)
