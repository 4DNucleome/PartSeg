from __future__ import division
import tifffile as tif
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QScrollArea, QPalette, QSizePolicy, QToolButton, QIcon, QSize, QAction, Qt, QPainter, QPen, \
    QColor, QScrollBar, QApplication, pyqtSignal, QPoint
from stack_settings import Settings

import matplotlib
from matplotlib import colors
import numpy as np
import os
from global_settings import file_folder, use_qt5
from math import log
step = 1.01
max_step = log(1.2, step)

canvas_icon_size = QSize(27, 27)


class MainMenu(QWidget):
    def __init__(self, settings):
        super(MainMenu, self).__init__()
        self.settings = settings
        self.load_btn = QPushButton("Load")
        self.save_btn = QPushButton("Save")
        layout = QHBoxLayout()
        layout.addWidget(self.load_btn)
        layout.addWidget(self.save_btn)
        self.setLayout(layout)

    def load_image(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setDirectory(self.settings.open_directory)
        filters = ["raw image (*.tiff *.tif *.lsm)"]
        dial.setNameFilters(filters)
        if not dial.exec_():
            return
        file_path = str(dial.selectedFiles()[0])
        self.settings.open_directory = os.path.dirname(str(file_path))
        im = tif.imread(file_path)
        self.settings.image = im


class ImageSettings(object):
    def __init__(self):
        self.zoom = False
        self.move = False

    def set_zoom(self, val):
        self.zoom = val

    def set_move(self, val):
        self.move = val


class ImageCanvas(QLabel):
    zoom_mark = pyqtSignal(QPoint, QPoint)

    def __init__(self, local_settings):
        """
        :type local_settings: ImageSettings
        :param local_settings: 
        """
        super(ImageCanvas, self).__init__()
        self.scale_factor = 1
        self.local_settings = local_settings
        self.point = None
        self.point2 = None

    def update_size(self, scale_factor=None):
        if scale_factor is not None:
            self.scale_factor = scale_factor
        self.resize(self.scale_factor * self.pixmap().size())

    def mousePressEvent(self, event):
        """
        :type event: QMouseEvent
        :param event: 
        :return: 
        """
        super(ImageCanvas, self).mousePressEvent(event)
        if self.local_settings.zoom:
            self.point = event.pos()

    def mouseMoveEvent(self, event):
        super(ImageCanvas, self).mouseMoveEvent(event)
        if self.local_settings.zoom:
            self.point2 = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        super(ImageCanvas, self).mouseReleaseEvent(event)
        if self.local_settings.zoom:
            self.zoom_mark.emit(self.point, self.point2)
            self.point2 = None
            self.update()

    def paintEvent(self, event):
        super(ImageCanvas, self).paintEvent(event)
        if not self.local_settings.zoom and self.point is None or self.point2 is None:
            return
        pen = QPen(QColor("white"))
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([5, 5])
        painter = QPainter(self)
        painter.setPen(pen)
        diff = self.point2 - self.point
        painter.drawRect(self.point.x(), self.point.y(), diff.x(), diff.y())
        pen = QPen(QColor("blue"))
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([5, 5])
        pen.setDashOffset(3)
        painter.setPen(pen)
        painter.drawRect(self.point.x(), self.point.y(), diff.x(), diff.y())

    """def resizeEvent(self, event):
        super(ImageCanvas, self).resizeEvent(event)
        print("Buka")
        event.accept()
        pass"""


def get_scroll_bar_proportion(scroll_bar):
    """
    :type scroll_bar: QScrollBar
    :param scroll_bar: 
    :return: float
    """
    dist = (scroll_bar.maximum() - scroll_bar.minimum())
    if dist == 0:
        return 0.5
    else:
        return float(scroll_bar.value()) / dist


def set_scroll_bar_proportion(scroll_bar, proportion):
    """
    :type scroll_bar: QScrollBar
    :type proportion: float
    :param scroll_bar: 
    """
    scroll_bar.setValue(int((scroll_bar.maximum() - scroll_bar.minimum())*proportion))


class MyScrollArea(QScrollArea):
    def __init__(self, local_settings, *args, **kwargs):
        """
        :type local_settings: ImageSettings
        :param local_settings: 
        :param args: 
        :param kwargs: 
        """
        super(MyScrollArea, self).__init__(*args, **kwargs)
        self.local_settings = local_settings
        self.setAlignment(Qt.AlignCenter)
        self.clicked = False
        self.prev_pos = None
        self.pixmap = ImageCanvas(local_settings)
        self.pixmap.setScaledContents(True)
        self.pixmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap.setBackgroundRole(QPalette.Base)
        self.pixmap.setScaledContents(True)
        self.pixmap.zoom_mark.connect(self.zoom_image)
        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(self.pixmap)
        self.ratio = 1

    def zoom_image(self, point1, point2):
        """
        :type point1: QPoint
        :type point2: QPoint
        :param point1: 
        :param point2: 
        :return: 
        """
        x_width = abs(point1.x() - point2.x())
        y_width = abs(point1.y() - point2.y())
        if x_width < 10 or y_width < 10:
            return
        x_ratio = self.width() / x_width
        y_ratio = self.height() / y_width
        scale_ratio = min(x_ratio, y_ratio)
        self.pixmap.resize(self.pixmap.size() * scale_ratio)
        img_h = self.pixmap.size().height()
        view_h = self.size().height() - 2
        y_mid = (point1.y() + point2.y())/2 * scale_ratio
        v_min = self.verticalScrollBar().minimum()
        v_max = self.verticalScrollBar().maximum()
        v_set = v_min + (v_max - v_min) * ((y_mid - view_h/2) / (img_h-view_h))
        self.verticalScrollBar().setValue(v_set)
        img_w = self.pixmap.size().width()
        view_w = self.size().width() - 2
        x_mid = (point1.x() + point2.x()) / 2 * scale_ratio
        v_min = self.horizontalScrollBar().minimum()
        v_max = self.horizontalScrollBar().maximum()
        v_set = v_min + (v_max - v_min) * ((x_mid - view_w / 2) / (img_w - view_w))
        self.horizontalScrollBar().setValue(v_set)

    def reset_image(self):
        x = self.size().width() - 2
        y = self.size().height() - 2
        if x > y * self.ratio:
            x = int(y * self.ratio)
        else:
            y = int(x / self.ratio)
        self.pixmap.resize(x, y)

    def set_image(self, im):
        width, height, _ = im.shape
        self.ratio = float(width)/float(height)
        im2 = QImage(im.data, width, height, im.dtype.itemsize * width * 4, QImage.Format_ARGB32)
        self.widget().setPixmap(QPixmap.fromImage(im2))
        self.widget().adjustSize()

    def mousePressEvent(self, event):
        self.clicked = True
        self.prev_pos = event.x(), event.y()

    def mouseReleaseEvent(self, event):
        self.clicked = False
        self.prev_pos = None

    def mouseMoveEvent(self, event):
        if not self.local_settings.move:
            return
        x, y = event.x(), event.y()
        x_dif, y_dif = self.prev_pos[0] - x, self.prev_pos[1] - y
        h_bar = self.horizontalScrollBar()
        h_bar.setValue(h_bar.value() + x_dif)
        v_bar = self.verticalScrollBar()
        v_bar.setValue(v_bar.value() + y_dif)
        self.prev_pos = x, y

    def showEvent(self, event):
        super(MyScrollArea, self).showEvent(event)
        if not event.spontaneous():
            scroll_bar = self.horizontalScrollBar()
            n_val = (scroll_bar.minimum() + scroll_bar.maximum()) / 2
            scroll_bar.setValue(n_val)
            scroll_bar = self.verticalScrollBar()
            n_val = (scroll_bar.minimum() + scroll_bar.maximum()) / 2
            scroll_bar.setValue(n_val)

    def resizeEvent(self, event):
        super(MyScrollArea, self).resizeEvent(event)
        if self.size().width() - 2 > self.pixmap.width() and self.size().height() - 2 > self.pixmap.height():
            self.reset_image()

    def wheelEvent(self, event):
        if not (QApplication.keyboardModifiers() & Qt.ControlModifier) == Qt.ControlModifier:
            return
        if use_qt5:
            delta = event.angleDelta().y()
        else:
            delta = event.delta()
        x, y = event.x(), event.y()
        if abs(delta) > max_step:
            delta = max_step * (delta/abs(delta))
        new_size = self.pixmap.size() * (step**delta)
        """:type : QSize"""
        if new_size.width() < self.size().width() - 2 and new_size.height() < self.size().height() - 2:
            self.reset_image()
        else:
            x0 = x - self.pixmap.x()
            y0 = y - self.pixmap.y()
            x_ratio = float(x0)/self.pixmap.width()
            y_ratio = float(y0)/self.pixmap.height()
            #scroll_h_ratio = get_scroll_bar_proportion(self.horizontalScrollBar())
            #scroll_v_ratio = get_scroll_bar_proportion(self.verticalScrollBar())
            self.pixmap.resize(new_size)
            set_scroll_bar_proportion(self.horizontalScrollBar(), y_ratio)
            set_scroll_bar_proportion(self.verticalScrollBar(), x_ratio)
        event.accept()


def create_tool_button(text, icon):
    res = QToolButton()
    res.setIconSize(canvas_icon_size)
    if icon is None:
        res.setText(text)
    else:
        res.setToolTip(text)
        if isinstance(icon, str):
            res.setIcon(QIcon(os.path.join(file_folder, "icons", icon)))
        else:
            res.setIcon(icon)
    return res


class ImageView(QWidget):
    def __init__(self):
        super(ImageView, self).__init__()
        self.local_settings = ImageSettings()
        self.btn_layout = QHBoxLayout()
        self.image_area = MyScrollArea(self.local_settings)
        self.reset_button = create_tool_button("Reset zoom", "zoom-original.png")
        self.reset_button.clicked.connect(self.reset_image_size)
        self.zoom_button = create_tool_button("Zoom", "zoom-select.png")
        self.zoom_button.clicked.connect(self.local_settings.set_zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        crop = QAction("Crop", self.zoom_button)
        # crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = create_tool_button("Move", "transform-move.png")
        self.move_button.clicked.connect(self.local_settings.set_move)
        self.move_button.setCheckable(True)
        self.btn_layout.addWidget(self.reset_button)
        self.btn_layout.addWidget(self.zoom_button)
        self.btn_layout.addWidget(self.move_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.btn_layout)
        main_layout.addWidget(self.image_area)
        self.setLayout(main_layout)

    def reset_image_size(self):
        self.image_area.reset_image()

    def set_image(self, image):
        self.image_area.set_image(image)

    def showEvent(self, event):
        super(ImageView, self).showEvent(event)
        if not event.spontaneous():
            self.btn_layout.addStretch(1)


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = Settings()
        self.main_menu = MainMenu(self.settings)

        self.scroll_area = ImageView()


        #self.scroll_area.setVisible(False)
        #self.scroll_area.setS

        im = tif.imread("stack.tif")
        width, height = im.shape
        im = colors.PowerNorm(gamma=1, vmin=im.min(), vmax=im.max())(im)
        cmap = matplotlib.cm.get_cmap("cubehelix")
        colored_image = cmap(im)
        # noinspection PyTypeChecker
        im = np.array(colored_image * 255, dtype=np.uint8)
        #im2 = QImage(im.data, width, height, im.dtype.itemsize*width*4, QImage.Format_ARGB32)

        #self.pixmap.setPixmap(QPixmap.fromImage(im2))
        #self.im_view = pg.ImageView(self)
        #self.im_view.setImage(im)
        layout = QVBoxLayout()
        layout.addWidget(self.main_menu)
        layout.addWidget(self.scroll_area)
        #self.pixmap.adjustSize()
        #self.pixmap.update_size(2)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        self.scroll_area.set_image(im)



