from __future__ import division, print_function
from qt_import import QMainWindow, QPixmap, QImage, QPushButton, QFileDialog, QWidget, QVBoxLayout, QHBoxLayout, \
    QLabel, QScrollArea, QPalette, QSizePolicy, QToolButton, QIcon, QSize, QAction, Qt, QPainter, QPen, QGridLayout, \
    QColor, QScrollBar, QApplication, pyqtSignal, QPoint, QSlider, QMessageBox, QCheckBox, QComboBox, QSize, QObject, \
    QEvent, QToolTip
import os
from global_settings import file_folder, use_qt5
from stack_settings import ImageSettings
from math import log
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.cm import get_cmap
from matplotlib import pyplot
import SimpleITK as sitk
import collections
from typing import Callable
import custom_colormaps

canvas_icon_size = QSize(27, 27)
step = 1.01
max_step = log(1.2, step)


class ImageState(QObject):
    parameter_changed = pyqtSignal()
    opacity_changed = pyqtSignal(float)
    show_label_changed = pyqtSignal(bool)

    def __init__(self):
        super(ImageState, self).__init__()
        self.zoom = False
        self.move = False
        self.opacity = 0.7
        self.show_label = True
        self.only_borders = True
        self.borders_thick = 2

    def set_zoom(self, val):
        self.zoom = val

    def set_move(self, val):
        self.move = val

    def set_borders(self, val):
        if self.only_borders != val:
            self.only_borders = val
            self.parameter_changed.emit()

    def set_borders_thick(self, val):
        if val != self.borders_thick:
            self.borders_thick = val
            self.parameter_changed.emit()

    def set_opacity(self, val):
        if self.opacity != val:
            self.opacity = val
            self.parameter_changed.emit()

    def set_show_label(self, val):
        if self.show_label != val:
            self.show_label = val
            self.parameter_changed.emit()


class ImageCanvas(QLabel):
    zoom_mark = pyqtSignal(QPoint, QPoint)
    position_signal = pyqtSignal(QPoint, QSize)
    click_signal = pyqtSignal(QPoint, QSize)
    leave_signal = pyqtSignal()

    def __init__(self, local_settings):
        """
        :type local_settings: ImageState
        :param local_settings:
        """
        super(ImageCanvas, self).__init__()
        self.scale_factor = 1
        self.local_settings = local_settings
        self.point = None
        self.point2 = None
        self.setMouseTracking(True)

    def update_size(self, scale_factor=None):
        if scale_factor is not None:
            self.scale_factor = scale_factor
        self.resize(self.scale_factor * self.pixmap().size())

    def leaveEvent(self, a0: QEvent):
        self.leave_signal.emit()

    def mousePressEvent(self, event):
        """
        :type event: QMouseEvent
        :param event:
        :return:
        """
        super(ImageCanvas, self).mousePressEvent(event)
        if self.local_settings.zoom:
            self.point = event.pos()
        elif not self.local_settings.move:
            self.click_signal.emit(event.pos(), self.size())

    def mouseMoveEvent(self, event):
        super(ImageCanvas, self).mouseMoveEvent(event)
        if self.local_settings.zoom and self.point is not None:
            self.point2 = event.pos()
            self.update()
        self.position_signal.emit(event.pos(), self.size())

    def mouseReleaseEvent(self, event):
        super(ImageCanvas, self).mouseReleaseEvent(event)
        if self.local_settings.zoom:
            diff = self.point2 - self.point
            if abs(diff.x()) and abs(diff.y()):
                self.zoom_mark.emit(self.point, self.point2)
            self.point2 = None
            self.point = None
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


default_colors = ['BlackRed', 'BlackGreen', 'BlackBlue', 'BlackMagenta']


class ChanelColor(QWidget):
    def __init__(self, num, *args, **kwargs):
        super(ChanelColor, self).__init__(*args, **kwargs)
        self.num = num
        self.check_box = QCheckBox(self)
        self.color_list = QComboBox(self)
        self.color_list.addItems(pyplot.colormaps())
        num2 = num % len(default_colors)
        pos = pyplot.colormaps().index(default_colors[num2])
        self.color_list.setCurrentIndex(pos)
        layout = QHBoxLayout()
        layout.addWidget(self.check_box)
        layout.addWidget(self.color_list)
        self.setLayout(layout)

    def channel_visible(self):
        return self.check_box.isChecked()

    def colormap(self, vmin, vmax):
        cmap = get_cmap(str(self.color_list.currentText()))
        norm = PowerNorm(1, vmin=vmin, vmax=vmax)
        return lambda x: cmap(norm(x))

    def register(self, fun):
        self.color_list.currentIndexChanged.connect(fun)
        self.check_box.stateChanged.connect(fun)

    def setVisible(self, val):
        super(ChanelColor, self).setVisible(val)
        self.check_box.setChecked(val)

    def set_list(self, colormap_list):
        """
        :type colormap_list: list[str]
        :param colormap_list:
        :return:
        """
        text = str(self.color_list.currentText())
        try:
            index = colormap_list.index(text)
        except ValueError as e:
            index = -1

        if index != -1:
            self.color_list.blockSignals(True)
        self.color_list.clear()
        self.color_list.addItems(text)
        if index != -1:
            self.color_list.setCurrentIndex(index)
            self.blockSignals(False)


class ImageView(QWidget):
    position_changed = pyqtSignal([int, int, int], [int, int])
    component_clicked = pyqtSignal(int)

    def __init__(self, settings):
        """:type settings: ImageSettings"""
        super(ImageView, self).__init__()
        self.exclude_btn_list = []
        self.check_fun = None
        self.image_state = ImageState()
        self.image_area = MyScrollArea(self.image_state)
        self.reset_button = create_tool_button("Reset zoom", "zoom-original.png")
        self.reset_button.clicked.connect(self.reset_image_size)
        self.zoom_button = create_tool_button("Zoom", "zoom-select.png")
        self.zoom_button.clicked.connect(self.image_state.set_zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.component = None
        crop = QAction("Crop", self.zoom_button)
        # crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = create_tool_button("Move", "transform-move.png")
        self.move_button.clicked.connect(self.image_state.set_move)
        self.move_button.setCheckable(True)
        self.chanel_color = [ChanelColor(x, self) for x in range(10)]
        self.info_text = QLabel()
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.reset_button)
        self.btn_layout.addWidget(self.zoom_button)
        self.btn_layout.addWidget(self.move_button)
        self.btn_layout.addStretch(1)
        self.btn_layout.addStretch(1)
        for el in self.chanel_color:
            self.btn_layout.addWidget(el)
            el.register(self.change_image)
        self.stack_slider = QSlider(Qt.Horizontal)
        self.stack_slider.valueChanged.connect(self.change_image)
        self.stack_slider.valueChanged.connect(self.change_layer)
        self.layer_info = QLabel()
        self.image = None
        self.channels_num = 1
        self.layers_num = 1
        self.border_val = []
        self.labels_layer = None
        self.image_shape = QSize(1, 1)

        main_layout = QVBoxLayout()
        main_layout.addLayout(self.btn_layout)
        main_layout.addWidget(self.image_area)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.stack_slider)
        slider_layout.addWidget(self.layer_info)
        main_layout.addLayout(slider_layout)
        main_layout.addWidget(self.info_text)

        self.setLayout(main_layout)
        self.exclude_btn_list.extend([self.zoom_button, self.move_button])
        self.zoom_button.clicked.connect(self.exclude_btn_fun)
        self.move_button.clicked.connect(self.exclude_btn_fun)

        self.image_state.parameter_changed.connect(self.change_image)
        self.image_area.pixmap.position_signal.connect(self.position_info)
        self.image_area.pixmap.leave_signal.connect(self.clean_text)
        self.image_area.pixmap.click_signal.connect(self.component_click)
        self.position_changed[int, int, int].connect(self.info_text_pos)
        self.position_changed[int, int].connect(self.info_text_pos)

    def set_check_fun(self, fun: Callable[[int], bool]):
        self.check_fun = fun

    def component_click(self, point, size):
        if self.labels_layer is None:
            return
        x = int(point.x() / size.width() * self.image_shape.width())
        y = int(point.y() / size.height() * self.image_shape.height())
        if self.layers_num > 1:
            num = self.labels_layer[self.stack_slider.value(), y, x]
        else:
            num = self.labels_layer[y, x]
        if num > 0:
            self.component_clicked.emit(num)

    def exclude_btn_fun(self):
        sender = self.sender()
        for el in self.exclude_btn_list:
            if el != sender:
                el.setChecked(False)

    def event(self, event: QEvent):

        if event.type() == QEvent.ToolTip and self.component is not None:
            text = str(self.component)
            if self.check_fun is not None:
                if self.check_fun(self.component):
                    text = "☑{}".format(self.component)
                else:
                    text = "☐{}".format(self.component)
            QToolTip.showText(event.globalPos(), text)
        return super(ImageView, self).event(event)

    def clean_text(self):
        self.info_text.setText("")

    def info_text_pos(self, *pos):
        brightness = self.image[pos]
        if isinstance(brightness, collections.Iterable):
            res_brightness = []
            for i, b in enumerate(brightness):
                if self.chanel_color[i].channel_visible():
                    res_brightness.append(b)
            brightness = res_brightness
            if len(brightness) == 1:
                brightness = brightness[0]
        if self.labels_layer is not None:
            comp = self.labels_layer[pos]
            self.component = comp
            if comp == 0:
                comp = "none"
                self.component = None
            self.info_text.setText("Position: {}, Brightness: {}, component {}".format(tuple(pos), brightness, comp))
        else:
            self.info_text.setText("Position: {}, Brightness: {}".format(tuple(pos), brightness))

    def position_info(self, point, size):
        """
        :type point: QPoint
        :type size: QSize
        :param point:
        :return:
        """
        x = int(point.x() / size.width() * self.image_shape.width())
        y = int(point.y() / size.height() * self.image_shape.height())
        if self.layers_num > 1:
            self.position_changed[int, int, int].emit(self.stack_slider.value()+1, y, x)
        else:
            self.position_changed[int, int].emit(y, x)

    def get_control_view(self):
        # type: () -> ImageState
        return self.image_state

    def reset_image_size(self):
        self.image_area.reset_image()

    def change_layer(self, num):
        self.layer_info.setText("{} of {}".format(num+1, self.layers_num))

    def change_image(self):
        if self.layers_num > 1:
            img = self.image[self.stack_slider.value()]
        else:
            img = self.image
        if self.channels_num > 1:
            res = np.zeros(img.shape[:2]+(4,), dtype=np.float)
            for i in range(self.channels_num):
                try:
                    if self.chanel_color[i].channel_visible():

                        colormap = self.chanel_color[i].colormap(*self.border_val[i])
                        res = np.maximum(res,  colormap(img[..., i]))
                except IndexError as e:
                    print(e)
                    break
        else:
            colormap = self.chanel_color[0].colormap(*self.border_val[0])
            res = colormap(img)
        # res[res > 1] = 1
        res = res[..., 0:3]
        im = np.array(res * 255, dtype=np.uint8)
        del res
        if self.labels_layer is not None and self.image_state.show_label:
            if self.layers_num > 1:
                layers = self.labels_layer[self.stack_slider.value()]
            else:
                layers = self.labels_layer
            if self.image_state.only_borders:
                bord = sitk.LabelContour(sitk.GetImageFromArray(layers))
                if self.image_state.borders_thick > 1:
                    bord = sitk.GrayscaleDilate(bord, self.image_state.borders_thick)
                labeled = sitk.GetArrayFromImage(sitk.LabelToRGB(bord))
                layers_mask = sitk.GetArrayFromImage(bord) > 0
            else:
                labeled = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(layers)))
                layers_mask = layers > 0
            im[layers_mask] = (1 - self.image_state.opacity) * im[layers_mask] + self.image_state.opacity * labeled[layers_mask]
        self.image_area.set_image(im, self.sender() is not None)

    def set_image(self, image):
        """
        :type image: np.ndarray
        :param image:
        :return:
        """
        self.channels_num = 1
        self.layers_num = 1
        self.border_val = []
        self.labels_layer = None
        image = np.squeeze(image)
        self.image = image
        if len(image.shape) == 2:
            self.image_shape = QSize(image.shape[1], image.shape[0])
            pass
            #self.image_area.set_image(image)
        elif len(image.shape) == 3:
            if image.shape[-1] < 10:
                self.channels_num = image.shape[-1]
                self.image_shape = QSize(image.shape[1], image.shape[0])
            else:
                self.layers_num = image.shape[0]
                self.image_shape = QSize(image.shape[2], image.shape[1])
        elif len(image.shape) == 4:
            self.channels_num = image.shape[-1]
            self.layers_num = image.shape[0]
            self.image_shape = QSize(image.shape[2], image.shape[1])
        else:
            QMessageBox.warning(self, "Open error", "Shape {} of image do not supported".format(image.shape))
        self.border_val = []
        if self.channels_num > 1:
            for i in range(self.channels_num):
                self.border_val.append((np.min(image[..., i]), np.max(image[..., i])))
        else:
            self.border_val = [(np.min(image), np.max(image))]
        self.stack_slider.blockSignals(True)
        self.stack_slider.setRange(0, self.layers_num - 1)
        self.stack_slider.setValue(int(self.layers_num/2))
        self.stack_slider.blockSignals(False)
        for el in self.chanel_color[self.channels_num:]:
            el.setVisible(False)
        for el in self.chanel_color[:self.channels_num]:
            el.setVisible(True)
        self.change_image()
        self.change_layer(int(self.layers_num/2))
        # self.image_area.set_image(image)

    def set_labels(self, labels):
        self.labels_layer = labels
        self.change_image()

    def showEvent(self, event):
        super(ImageView, self).showEvent(event)
        #if not event.spontaneous():
        #    self.btn_layout.addStretch(1)


class MyScrollArea(QScrollArea):
    """
    :type image_ratio: float
    :param image_ratio: image width/height ratio
    :type zoom_scale: float
    :param zoom_scale: zoom scale
    """
    resize_area = pyqtSignal(QSize)

    def __init__(self, local_settings, *args, **kwargs):
        """
        :type local_settings: ImageState
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
        self.image_ratio = 1
        self.zoom_scale = 1
        self.max_zoom = 10
        self.image_size = QSize(1, 1)
        self.resize_area.connect(self.pixmap.resize)

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
        self.zoom_scale *= scale_ratio
        if self.zoom_scale > self.max_zoom:
            scale_ratio *= self.max_zoom / self.zoom_scale
            self.zoom_scale = self.max_zoom
        print(self.zoom_scale)
        if scale_ratio == 1:
            return
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
        if float(x) > y * self.image_ratio:
            x = int(y * self.image_ratio)
        else:
            y = int(x / self.image_ratio)
        self.pixmap.resize(x, y)
        self.zoom_scale = x/self.image_size.width()

    def set_image(self, im, keep_size=False):
        height, width, _ = im.shape
        self.image_size = QSize(width, height)
        self.image_ratio = float(width) / float(height)
        im2 = QImage(im.data, width, height, im.dtype.itemsize * width * 3, QImage.Format_RGB888)
        self.widget().setPixmap(QPixmap.fromImage(im2))
        if not keep_size:
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
        scale_mod = (step**delta)
        if self.zoom_scale * scale_mod > self.max_zoom:
            scale_mod = self.max_zoom / self.zoom_scale

        new_size = self.pixmap.size() * scale_mod
        if scale_mod == 1:
            return
        self.zoom_scale *= scale_mod
        """:type : QSize"""
        if new_size.width() < self.size().width() - 2 and new_size.height() < self.size().height() - 2:
            self.reset_image()
        else:
            x0 = x - self.pixmap.x()
            y0 = y - self.pixmap.y()
            x_ratio = float(x0)/self.pixmap.width()
            y_ratio = float(y0)/self.pixmap.height()
            # scroll_h_ratio = get_scroll_bar_proportion(self.horizontalScrollBar())
            # scroll_v_ratio = get_scroll_bar_proportion(self.verticalScrollBar())
            self.resize_area.emit(new_size)
            set_scroll_bar_proportion(self.horizontalScrollBar(), y_ratio)
            set_scroll_bar_proportion(self.verticalScrollBar(), x_ratio)
        event.accept()