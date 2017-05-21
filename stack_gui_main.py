import tifffile as tif
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QScrollArea, QPalette, QSizePolicy, QToolButton, QIcon, QSize, QAction, Qt
from stack_settings import Settings

import matplotlib
from matplotlib import colors
import numpy as np
import os

from global_settings import file_folder

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


class ImageCanvas(QLabel):
    def __init__(self):
        super(ImageCanvas, self).__init__()
        self.scale_factor = 1

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

    def mouseReleaseEvent(self, event):
        super(ImageCanvas, self).mouseReleaseEvent(event)


class MyScrollArea(QScrollArea):
    def __init__(self, *args, **kwargs):
        super(MyScrollArea, self).__init__(*args, **kwargs)
        self.setAlignment(Qt.AlignCenter)
        self.clicked = False
        self.prev_pos = None
        self.pixmap = ImageCanvas()
        self.pixmap.setScaledContents(True)
        self.pixmap.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.pixmap.setBackgroundRole(QPalette.Base)
        self.pixmap.setScaledContents(True)
        self.setBackgroundRole(QPalette.Dark)
        self.setWidget(self.pixmap)
        self.ratio = 1

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
        self.btn_layout = QHBoxLayout()
        self.image_area = MyScrollArea()
        self.reset_button = create_tool_button("Reset zoom", "zoom-original.png")
        self.reset_button.clicked.connect(self.reset_image_size)
        self.zoom_button = create_tool_button("Zoom", "zoom-select.png")
        # self.zoom_button.clicked.connect(self.zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        crop = QAction("Crop", self.zoom_button)
        # crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = create_tool_button("Move", "transform-move.png")
        # self.move_button.clicked.connect(self.move_action)
        self.move_button.setCheckable(True)
        self.btn_layout.addWidget(self.reset_button)
        self.btn_layout.addWidget(self.zoom_button)
        self.btn_layout.addWidget(self.move_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.btn_layout)
        main_layout.addWidget(self.image_area)
        self.setLayout(main_layout)

    def reset_image_size(self):
        print(self.image_area.size())
        self.image_area.reset_image()
        # self.image_area.setWidgetResizable(True)

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

        im = tif.imread("data/A.lsm")[0, 20, 3]
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



