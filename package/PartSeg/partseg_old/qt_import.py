# coding=utf-8
import logging
import platform
import sys

import matplotlib

from partseg_utils.global_settings import use_qt5, button_margin, button_small_dist

__author__ = "Grzegorz Bokota"



# use_qt5 = False


import matplotlib as mpl
print(mpl.get_cachedir())


if use_qt5:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

    from PyQt5.QtCore import Qt, QSize, QTimer, QVariant, pyqtSignal, QObject, QPoint, QRect, QEvent, QThread
    from PyQt5.QtWidgets import QLabel, QPushButton, QFileDialog, QMainWindow, QStatusBar, QWidget, \
        QLineEdit, QFrame,  QMessageBox, QSlider, QCheckBox, QComboBox, QSpinBox, QToolButton, QDoubleSpinBox, \
        QAbstractSpinBox, QApplication, QTabWidget, QScrollArea, QInputDialog, QHBoxLayout, QVBoxLayout, QListWidget, \
        QTextEdit, QDialog, QTableWidget, QTableWidgetItem, QGridLayout, QAction, QListWidgetItem, QDockWidget, \
        QTextBrowser, QSplitter, QProgressBar, QAbstractItemView, QGroupBox, QTreeWidget, QTreeWidgetItem, QCompleter, \
        QRadioButton, QButtonGroup, QSizePolicy, QScrollBar, QLayout, QLayoutItem, QStyle, QFormLayout, \
        QStackedLayout, QToolTip

    from PyQt5.QtGui import QFont, QFontMetrics, QIcon, QPixmap, QImage, QPainter, QPen, QColor, QPalette, QHelpEvent
    from PyQt5.QtHelp import QHelpEngine
else:
    raise NotImplementedError()


# from http://stackoverflow.com/questions/5671354/how-to-programmatically-make-a-horizontal-line-in-qt

if sys.version_info.major == 2:
    str_type = unicode
else:
    str_type = str


def h_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.HLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto


def v_line():
    toto = QFrame()
    toto.setFrameShape(QFrame.VLine)
    toto.setFrameShadow(QFrame.Sunken)
    return toto




def set_position(elem, previous, dist=10):
    pos_y = previous.pos().y()
    if platform.system() == "Darwin" and isinstance(elem, QLineEdit):
        pos_y += 3
    if platform.system() == "Darwin" and isinstance(previous, QLineEdit):
        pos_y -= 3
    if platform.system() == "Darwin" and isinstance(previous, QSlider):
        pos_y -= 10
    if platform.system() == "Darwin" and isinstance(elem, QSpinBox):
        pos_y += 7
    if platform.system() == "Darwin" and isinstance(previous, QSpinBox):
        pos_y -= 7
    elem.move(previous.pos().x() + previous.size().width() + dist, pos_y)


def set_button(button, previous_element, dist=10, super_space=0):
    """
    :type button: QPushButton | QLabel
    :type previous_element: QWidget | None
    :param button:
    :param previous_element:
    :param dist:
    :param super_space:
    :return:
    """
    font_met = QFontMetrics(button.font())
    if isinstance(button, QComboBox):
        text_list = [button.itemText(i) for i in range(button.count())]
    else:
        text = button.text()
        if text[0] == '&':
            text = text[1:]
        text_list = text.split("\n")
    if isinstance(button, QSpinBox):
        button.setAlignment(Qt.AlignRight)
        text_list = [str(button.maximum()) + 'aa']
        # print(text_list)
    width = 0
    for txt in text_list:
        width = max(width, font_met.boundingRect(txt).width())
    if isinstance(button, QPushButton):
        button.setFixedWidth(width + button_margin + super_space)
    if isinstance(button, QLabel):
        button.setFixedWidth(width + super_space)
    if isinstance(button, QComboBox):
        button.setFixedWidth(width + button_margin + 10)
    if isinstance(button, QSpinBox):
        # print(width)
        button.setFixedWidth(width)
    # button.setFixedHeight(button_height)
    if isinstance(previous_element, QCheckBox):
        dist += 20
    if previous_element is not None:
        set_position(button, previous_element, dist)


def pack_layout(*args):
    layout = QHBoxLayout()
    layout.setSpacing(0)
    layout.setContentsMargins(0, 0, 0, 0)
    for el in args:
        layout.addWidget(el)
    return layout