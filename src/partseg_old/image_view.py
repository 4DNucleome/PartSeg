import logging
import os

import SimpleITK as sitk
import matplotlib
import numpy as np
from matplotlib import colors
from matplotlib import pyplot

from partseg_old.backend import GAUSS
from partseg_old.segment import SegmentationProfile
from partseg_utils.global_settings import static_file_folder, develop
from partseg_utils.image_operations import DrawType
from partseg_old.qt_import import QWidget, FigureCanvas, QToolButton, QSize, QIcon, QAction, QLabel, QDialog, NavigationToolbar, \
    Qt, QSlider, QCheckBox, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QGridLayout, QInputDialog, \
    QApplication, QImage, QPixmap

canvas_icon_size = QSize(27, 27)


def label_to_rgb(image):
    sitk_im = sitk.GetImageFromArray(image)
    lab_im = sitk.LabelToRGB(sitk_im)
    return sitk.GetArrayFromImage(lab_im)


class ImageView(QWidget):
    def __init__(self):
        super(ImageView, self).__init__()
        self.pixmap = None
        self.label_pixmap = QLabel()

    def draw_image(self, image):
        """
        :type image: np.ndarray
        :param image: ARGB 8bit image 
        :return: 
        """
        width, height = image.shape
        im = QImage(image.data, width, height, image.dtype.itemsize * width * 4, QImage.Format_ARGB32)
        self.pixmap = QPixmap.fromImage(im)
        self.label_pixmap.setPixmap(self.pixmap)

    def paint_pixel(self):
        pass


class MyCanvas(QWidget):
    """
    :type settings: Settings
    """
    def __init__(self, figure_size, settings, info_object, parent, settings_callback=True):
        """
        Create basic canvas to view image
        :param figure_size: Size of figure in inches
        """

        fig = pyplot.figure(figsize=figure_size, dpi=100, frameon=False, facecolor='1.0', edgecolor='w',
                            tight_layout=True)
        # , tight_layout=tight_dict)
        super(MyCanvas, self).__init__(parent)
        self.settings = settings
        self.info_object = info_object

        self.figure_canvas = FigureCanvas(fig)
        self.base_image = None
        self.gauss_image = None
        self.max_value = 1
        self.min_value = 0
        self.ax_im = None
        self.rgb_image = None
        self.layer_num = 0
        self.main_layout = None
        self.zoom_sync = False
        self.sync_fig_num = 0
        self.rendered_image = []
        # self.setParent(parent)
        self.my_figure_num = fig.number
        self.toolbar = NavigationToolbar(self.figure_canvas, self)
        self.toolbar.hide()
        self.reset_button = QToolButton(self)
        self.reset_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "zoom-original.png")))
        self.reset_button.setIconSize(canvas_icon_size)
        self.reset_button.clicked.connect(self.reset)
        self.reset_button.setToolTip("Reset zoom")
        self.zoom_button = QToolButton(self)
        self.zoom_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "zoom-select.png")))
        self.zoom_button.setIconSize(canvas_icon_size)
        self.zoom_button.setToolTip("Zoom")
        self.zoom_button.clicked.connect(self.zoom)
        self.zoom_button.setCheckable(True)
        self.zoom_button.setContextMenuPolicy(Qt.ActionsContextMenu)
        crop = QAction("Crop", self.zoom_button)
        crop.triggered.connect(self.crop_view)
        self.zoom_button.addAction(crop)
        self.move_button = QToolButton(self)
        self.move_button.setToolTip("Move")
        self.move_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "transform-move.png")))
        self.move_button.setIconSize(canvas_icon_size)
        self.move_button.clicked.connect(self.move_action)
        self.move_button.setCheckable(True)
        # self.back_button = QPushButton("Undo", self)
        # noinspection PyUnresolvedReferences
        # self.back_button.clicked.connect(self.toolbar.back)
        # self.next_button = QPushButton("Redo", self)
        # self.next_button.clicked.connect(self.toolbar.forward)
        self.button_list = [self.reset_button, self.zoom_button, self.move_button] #, self.back_button, self.next_button]
        self.mouse_pressed = False
        self.begin_pos = None
        self.last_pos = None

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 0)
        self.slider.valueChanged[int].connect(self.change_layer)
        self.colormap_checkbox = QCheckBox(self)
        self.colormap_checkbox.setText("With colormap")
        self.colormap_checkbox.setChecked(True)
        self.mark_mask = QCheckBox(self)
        self.mark_mask.setText("Mark mask")
        self.mark_mask.setChecked(False)
        self.gauss_view = QCheckBox(self)
        self.gauss_view.setText("Gauss image")
        self.gauss_view.setChecked(False)
        # self.mark_mask.setDisabled(True)
        self.layer_num_label = QLabel(self)
        self.layer_num_label.setText("1 of 1      ")
        self.colormap_checkbox.stateChanged.connect(self.update_colormap)
        self.mark_mask.stateChanged.connect(self.update_colormap)
        self.gauss_view.stateChanged.connect(self.update_colormap)
        if settings_callback:
            settings.add_image_callback((self.set_image, GAUSS))
            settings.add_colormap_callback(self.update_colormap)
            self.settings.add_threshold_type_callback(self.update_colormap)
        # MyCanvas.update_elements_positions(self)
        # self.setFixedWidth(500)
        self.figure_canvas.mpl_connect('button_release_event', self.zoom_sync_fun)
        self.figure_canvas.mpl_connect('button_release_event', self.move_sync_fun)
        self.figure_canvas.mpl_connect('button_release_event', self.mouse_up)
        self.figure_canvas.mpl_connect('button_press_event', self.mouse_down)
        self.figure_canvas.mpl_connect('motion_notify_event', self.brightness_up)
        self.figure_canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.figure_canvas.mpl_connect('scroll_event', self.zoom_scale)

    def crop_view(self):
        shape = self.base_image.shape
        shape = shape[-1], shape[-2]
        fig = pyplot.figure(self.my_figure_num)
        xlim = pyplot.xlim()
        ylim = pyplot.ylim()
        cr = CropSet(shape, ((xlim[0] + 0.5, xlim[1] + 0.5), (ylim[1] + 0.5, ylim[0] + 0.5)))
        if cr.exec_():
            res = cr.get_range()
            logging.debug("crop {}".format(res))
            pyplot.xlim(res[0][0] - 0.5, res[0][1] - 0.5)
            pyplot.ylim(res[1][1] - 0.5, res[1][0] - 0.5)
            fig.canvas.draw()

    def zoom_scale(self, event):
        if self.zoom_button.isChecked() or self.move_button.isChecked():
            scale_factor = self.settings.scale_factor
            if (QApplication.keyboardModifiers() & Qt.ControlModifier) == Qt.ControlModifier:
                scale_factor -= (1 - scale_factor) * 2
            if event.button == "down":
                scale_factor = 1 / scale_factor

            def new_pos(mid, pos):
                return mid - (mid - pos) * scale_factor

            fig = pyplot.figure(self.my_figure_num)
            ax_size = pyplot.xlim()
            ay_size = pyplot.ylim()
            ax_size_n = (new_pos(event.xdata, ax_size[0]), new_pos(event.xdata, ax_size[1]))
            ay_size_n = (new_pos(event.ydata, ay_size[0]), new_pos(event.ydata, ay_size[1]))

            pyplot.xlim(ax_size_n)
            pyplot.ylim(ay_size_n)
            fig.canvas.draw()
            if self.zoom_sync:
                fig = pyplot.figure(self.sync_fig_num)
                pyplot.xlim(ax_size_n)
                pyplot.ylim(ay_size_n)
                fig.canvas.draw()

    def brightness_up(self, event):
        if self.info_object is None:
            return
        if event.xdata is not None and event.ydata is not None:
            x, y = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if self.gauss_view.isChecked():
                img = self.gauss_image
            else:
                img = self.base_image
            """:type img: np.ndarray"""
            try:
                if img.ndim == 2:
                    self.info_object.update_brightness(img[y, x], (x, y))
                else:
                    self.info_object.update_brightness(img[self.layer_num, y, x], (x,y))
            except IndexError:
                pass
        else:
            self.info_object.update_brightness(None)

    def mouse_up(self, _):
        self.mouse_pressed = False

    def mouse_down(self, event):
        self.mouse_pressed = True
        if event.xdata is not None and event.ydata is not None:
            self.begin_pos = event.xdata, event.ydata + 0.5
            self.last_pos = self.begin_pos
        else:
            self.begin_pos = None
            self.last_pos = None

    def mouse_move(self, event):
        if self.last_pos is None:
            return
        if event.xdata is not None:
            self.last_pos = event.xdata, self.last_pos[1]
        if event.ydata is not None:
            self.last_pos = self.last_pos[0], event.ydata

    def zoom_sync_fun(self, _):
        if self.zoom_sync and self.zoom_button.isChecked():
            fig = pyplot.figure(self.sync_fig_num)
            x_size = self.begin_pos[0], self.last_pos[0]
            if x_size[0] > x_size[1]:
                x_size = x_size[1], x_size[0]
            y_size = self.begin_pos[1], self.last_pos[1]
            if y_size[0] < y_size[1]:
                y_size = y_size[1], y_size[0]
            if abs(x_size[0] - x_size[1]) < 3 or abs(y_size[0] - y_size[1]) < 3:
                return
            pyplot.xlim(x_size)
            pyplot.ylim(y_size)
            fig.canvas.draw()

    def move_sync_fun(self, _):
        if self.zoom_sync and self.move_button.isChecked():
            pyplot.figure(self.my_figure_num)
            ax_size = pyplot.xlim()
            ay_size = pyplot.ylim()
            fig = pyplot.figure(self.sync_fig_num)
            pyplot.xlim(ax_size)
            pyplot.ylim(ay_size)
            fig.canvas.draw()

    def update_elements_positions(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)
        button_layout.setContentsMargins(0, 0, 0, 0)
        for butt in self.button_list:
            button_layout.addWidget(butt)
        button_layout.addStretch()
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.figure_canvas)
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setContentsMargins(0, 0, 0, 0)
        checkbox_layout.addWidget(self.colormap_checkbox)
        checkbox_layout.addWidget(self.mark_mask)
        checkbox_layout.addWidget(self.gauss_view)
        checkbox_layout.setSpacing(10)
        checkbox_layout.addStretch()
        main_layout.addLayout(checkbox_layout)
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addSpacing(15)
        slider_layout.addWidget(self.layer_num_label)
        main_layout.addLayout(slider_layout)
        self.setLayout(main_layout)
        self.main_layout = main_layout
        # print(self.__class__.__name__, "Spacing", main_layout.spacing(), button_layout.spacing())

    def sync_zoom(self, state):
        self.zoom_sync = state

    def move_action(self):
        """
        Deactivate zoom button and call function witch allow moving image
        :return: None
        """
        if self.zoom_button.isChecked():
            self.zoom_button.setChecked(False)
        self.toolbar.pan()

    def zoom(self):
        """
        Deactivate move button and call function witch allow moving image
        :return: None
        """
        if self.move_button.isChecked():
            self.move_button.setChecked(False)
        self.toolbar.zoom()

    def reset(self):
        # self.toolbar.home()
        fig = pyplot.figure(self.my_figure_num)
        ax_size = (-0.5, self.base_image.shape[-1] - 0.5)
        ay_size = (self.base_image.shape[-2] - 0.5, -0.5)
        pyplot.xlim(ax_size)
        pyplot.ylim(ay_size)
        fig.canvas.draw()
        if self.zoom_sync:
            fig = pyplot.figure(self.sync_fig_num)
            pyplot.xlim(ax_size)
            pyplot.ylim(ay_size)
            fig.canvas.draw()

    def update_image(self):
        self.rendered_image = [None] * self.settings.image.shape[0]

    def update_gauss(self):
        if self.gauss_view.isChecked():
            self.rendered_image = [None] * self.settings.image.shape[0]

    def set_image(self, image, gauss, image_update=False):
        """
        :type image: np.ndarray
        :type gauss: np.ndarray
        :type image_update: bool
        :return:
        """
        self.base_image = image
        pyplot.figure(self.my_figure_num)
        ax_lim = pyplot.xlim()
        ay_lim = pyplot.ylim()
        self.max_value = image.max()
        self.min_value = image.min()
        self.gauss_image = gauss
        if gauss is None:
            self.gauss_view.setDisabled(True)
        else:
            self.gauss_view.setEnabled(True)
        self.ax_im = None
        self.update_rgb_image()

        if len(image.shape) > 2:
            val = self.slider.value()
            self.slider.setRange(0, image.shape[0] - 1)
            new_val = int(image.shape[0] / 2)
            self.slider.setValue(new_val)
            if val == new_val:
                self.update_image_view()
        else:
            self.update_image_view()
        if image_update:
            pyplot.xlim(ax_lim)
            pyplot.ylim(ay_lim)

    def update_colormap(self):
        if self.base_image is None:
            return
        self.update_rgb_image()
        self.update_image_view()

    @property
    def image(self):
        """
        :return: np.ndarray
        """
        if len(self.image.shape) == 2:
            if self.gauss_view.isChecked():
                return self.gauss_image
            else:
                return self.base_image
        else:
            if self.gauss_view.isChecked():
                return self.gauss_image[self.layer_num]
            else:
                return self.base_image[self.layer_num]

    def update_rgb_image(self):
        if not self.settings.normalize_range[2]:
            norm = colors.PowerNorm(gamma=self.settings.power_norm,
                                    vmin=self.min_value, vmax=self.max_value)
        else:
            norm = colors.PowerNorm(gamma=self.settings.power_norm,
                                    vmin=self.settings.normalize_range[0],
                                    vmax=self.settings.normalize_range[1])
        if self.gauss_view.isChecked():
            float_image = norm(self.gauss_image)
        else:
            float_image = norm(self.base_image)
        if self.mark_mask.isChecked() and self.settings.mask is not None:
            zero_mask = self.settings.mask == 0
            mean_val = np.mean(float_image[zero_mask])
            if mean_val < 0.5:
                float_image[zero_mask] = \
                    (1 - self.settings.mask_overlay) * float_image[zero_mask] + self.settings.mask_overlay
            else:
                float_image[zero_mask] = \
                    (1 - self.settings.mask_overlay) * float_image[zero_mask] + \
                    self.settings.mask_overlay * float_image.min()
        if self.colormap_checkbox.isChecked():
            cmap = self.settings.color_map
        else:
            cmap = matplotlib.cm.get_cmap("gray")
        colored_image = cmap(float_image)
        self.rgb_image = np.array(colored_image * 255, dtype=np.uint8)

    def change_layer(self, layer_num):
        self.layer_num = layer_num
        self.layer_num_label.setText("{0} of {1}".format(str(layer_num + 1), str(self.base_image.shape[0])))
        self.update_image_view()

    def update_image_view(self):
        pyplot.figure(self.my_figure_num)
        if self.base_image.size < 10:
            return
        if len(self.base_image.shape) <= 2:
            image_to_show = self.rgb_image
        else:
            image_to_show = self.rgb_image[self.layer_num]
        if self.ax_im is None:
            pyplot.clf()
            self.ax_im = pyplot.imshow(image_to_show, interpolation='nearest')
        else:
            self.ax_im.set_data(image_to_show)
        self.figure_canvas.draw()

    """def resizeEvent(self, resize_event):
        super(MyCanvas, self).resizeEvent(resize_event)
        # print(self.__class__.__name__, self.size(), resize_event.size(), resize_event.oldSize())
        # self.update_elements_positions()
        self.updateGeometry()"""

    def get_image(self):
        if len(self.base_image.shape) <= 2:
            image_to_show = self.rgb_image
        else:
            image_to_show = self.rgb_image[self.layer_num]
        pyplot.figure(self.my_figure_num)
        return image_to_show, pyplot.xlim(), pyplot.ylim()


class MyDrawCanvas(MyCanvas):
    """
    :type segmentation: np.ndarray
    :type segment: Segment
    """

    def __init__(self, figure_size, settings, info_object, segment, *args):
        super(MyDrawCanvas, self).__init__(figure_size, settings, info_object, *args)
        self.draw_canvas = DrawObject(settings, segment, self.draw_update)
        self.history_list = list()
        self.redo_list = list()

        self.draw_button = QToolButton(self)
        self.draw_button.setToolTip("Draw")
        self.draw_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "draw-path.png")))
        self.draw_button.setIconSize(canvas_icon_size)
        self.draw_button.setCheckable(True)
        # self.draw_button.clicked[bool].connect(self.draw_click)
        self.erase_button = QToolButton(self)
        self.erase_button.setToolTip("Erase")
        self.erase_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "draw-eraser.png")))
        self.erase_button.setIconSize(canvas_icon_size)
        self.erase_button.setCheckable(True)
        # self.erase_button.clicked[bool].connect(self.erase_click)
        self.scale_button = QToolButton(self)
        self.scale_button.setToolTip("x, y image scale")
        self.scale_button.setIcon(QIcon(os.path.join(static_file_folder, "icons", "transform-scale.png")))
        self.scale_button.setIconSize(canvas_icon_size)
        self.scale_button.clicked.connect(self.scale_image)
        if not develop:
            self.scale_button.hide()
        self.show_button = QPushButton("Show", self)
        self.hide_button = QPushButton("Hide", self)
        self.show_button.setCheckable(True)
        self.hide_button.setCheckable(True)
        self.zoom_button.clicked.connect(self.generate_up_button_fun(self.zoom_button))
        self.move_button.clicked.connect(self.generate_up_button_fun(self.move_button))
        self.draw_button.clicked.connect(self.generate_up_button_fun(self.draw_button))
        self.erase_button.clicked.connect(self.generate_up_button_fun(self.erase_button))
        self.show_button.clicked.connect(self.generate_up_button_fun(self.show_button))
        self.hide_button.clicked.connect(self.generate_up_button_fun(self.hide_button))
        self.draw_button.clicked[bool].connect(self.generate_draw_modify(DrawType.draw))
        self.erase_button.clicked[bool].connect(self.generate_draw_modify(DrawType.erase))
        self.show_button.clicked[bool].connect(self.generate_draw_modify(DrawType.force_show))
        self.hide_button.clicked[bool].connect(self.generate_draw_modify(DrawType.force_hide))
        self.clean_button = QPushButton("Clean drawing", self)
        self.clean_button.clicked.connect(self.draw_canvas.clean)
        self.clean_data_button = QPushButton("Clean data", self)
        self.clean_data_button.clicked.connect(self.action_choose)
        self.button_list.extend([self.draw_button, self.erase_button, self.scale_button, self.show_button, self.hide_button,
                                 self.clean_button, self.clean_data_button])
        self.segment = segment
        self.protect_button = False
        self.segmentation = None
        self.rgb_segmentation = None
        self.original_rgb_image = None
        self.labeled_rgb_image = None
        self.cursor_val = 0
        self.colormap_checkbox.setChecked(False)
        self.segment.add_segmentation_callback(self.segmentation_changed)
        self.slider.valueChanged[int].connect(self.draw_canvas.set_layer_num)
        self.slider.valueChanged[int].connect(self.settings.change_layer)
        self.figure_canvas.mpl_connect('button_press_event', self.draw_canvas.on_mouse_down)
        self.figure_canvas.mpl_connect('motion_notify_event', self.draw_canvas.on_mouse_move)
        self.figure_canvas.mpl_connect('button_release_event', self.draw_canvas.on_mouse_up)

    def scale_image(self):
        dial = QInputDialog()
        val, ok = QInputDialog.getDouble(self, "Scale factor", "Set scale factor", 1, 0.01, 3, 2)
        if ok :
            print("Buka {}".format(val))

    def action_choose(self):
        if str(self.clean_data_button.text()) == "Clean data":
            self.remove_noise()
            self.clean_data_button.setText("Restore image")
        else:
            self.restore_original_image()
            self.clean_data_button.setText("Clean data")

    def restore_original_image(self):
        self.settings.image = self.settings.original_image
        self.settings.image_changed_fun()
        self.settings.image_clean_profile = None

    def remove_noise(self):
        image = np.copy(self.settings.image)
        mask = self.segment.get_segmentation()
        full_mask = self.segment.get_full_segmentation()
        if not np.any(np.array(full_mask == 0)):
            return
        noise_mean = np.mean(image[full_mask == 0])
        noise_mask = np.copy(full_mask)
        # noise_std = np.std(image[full_mask == 0])
        # noise_generator = norm(noise_mean, noise_std)
        noise_mask[mask > 0] = 0
        profile = SegmentationProfile("Clean_profile", **self.settings.get_profile_dict())
        self.settings.image_clean_profile = profile
        image[noise_mask > 0] = noise_mean
        # image[noise_mask > 0] = noise_generator.rvs(np.count_nonzero(noise_mask))
        self.settings.image = image
        self.settings.image_changed_fun()

    def draw_update(self, view=True):
        if view:
            self.update_image_view()
        x = int(self.draw_canvas.f_x)
        y = int(self.draw_canvas.f_y)
        segmentation = self.segment.get_segmentation()
        if len(segmentation.shape) == 3:
            val = segmentation[self.layer_num, y, x]
        else:
            val = segmentation[y, x]
        if val == self.cursor_val:
            return
        else:
            self.cursor_val = val
        if val == 0:
            self.info_object.update_info_text("No component")
        else:
            size = self.segment.get_size_array()[val]
            self.info_object.update_info_text("Component: {}, size: {}".format(val, size))

    def generate_up_button_fun(self, button):
        butt_list = [self.zoom_button, self.move_button, self.draw_button, self.erase_button, self.show_button,
                     self.hide_button]
        butt_list.remove(button)

        def fin_fun():
            if self.protect_button:
                return
            self.protect_button = True
            for butt in butt_list:
                if butt.isChecked():
                    butt.click()
            self.protect_button = False

        return fin_fun

    def up_move_zoom_button(self):
        self.protect_button = True
        if self.zoom_button.isChecked():
            self.zoom_button.click()
        if self.move_button.isChecked():
            self.move_button.click()
        self.protect_button = False

    def up_drawing_button(self):
        # TODO Update after create draw object
        if self.protect_button:
            return
        self.draw_canvas.draw_on = False
        if self.draw_button.isChecked():
            self.draw_button.setChecked(False)
        if self.erase_button.isChecked():
            self.erase_button.setChecked(False)

    def generate_draw_modify(self, t):
        def draw_modify(checked):
            if checked:
                self.draw_canvas.draw_on = True
                self.draw_canvas.value = t.value
            else:
                self.draw_canvas.draw_on = False
        return draw_modify

    def draw_click(self, checked):
        if checked:
            self.erase_button.setChecked(False)
            self.draw_canvas.set_draw_mode()
        else:
            self.draw_canvas.draw_on = False

    def erase_click(self, checked):
        if checked:
            self.draw_button.setChecked(False)
            self.draw_canvas.set_erase_mode()
        else:
            self.draw_canvas.draw_on = False

    def update_elements_positions(self):
        super(MyDrawCanvas, self).update_elements_positions()

    def segmentation_changed(self):
        self.update_segmentation_rgb()
        self.update_image_view()

    def update_segmentation_rgb(self):
        if self.original_rgb_image is None:
            self.update_rgb_image()
        self.update_segmentation_image()
        mask = self.segmentation > 0
        overlay = self.settings.overlay
        self.rgb_image = np.copy(self.original_rgb_image)
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + self.rgb_segmentation[mask] * overlay
        self.labeled_rgb_image = np.copy(self.rgb_image)
        draw_lab = label_to_rgb(self.draw_canvas.draw_canvas)
        if self.settings.use_draw_result:
            mask = (self.draw_canvas.draw_canvas == 1)
            mask *= (self.segmentation == 0)
        else:
            mask = self.draw_canvas.draw_canvas > 0
        self.rgb_image[mask] = self.original_rgb_image[mask] * (1 - overlay) + draw_lab[mask] * overlay
        self.draw_canvas.rgb_image = self.rgb_image
        self.draw_canvas.original_rgb_image = self.original_rgb_image
        self.draw_canvas.labeled_rgb_image = self.labeled_rgb_image

    def update_rgb_image(self):
        super(MyDrawCanvas, self).update_rgb_image()
        self.rgb_image = self.rgb_image[..., :3]
        self.original_rgb_image = np.copy(self.rgb_image)
        self.update_segmentation_rgb()

    def set_image(self, image, gauss, image_update=False):
        self.base_image = image
        pyplot.figure(self.my_figure_num)
        ax_lim = pyplot.xlim()
        ay_lim = pyplot.ylim()
        self.max_value = image.max()
        self.min_value = image.min()
        self.gauss_image = gauss
        self.ax_im = None
        self.original_rgb_image = None
        self.draw_canvas.set_image(image)
        self.segment.set_image()
        self.update_rgb_image()
        if len(image.shape) > 2:
            self.slider.setRange(0, image.shape[0] - 1)
            self.slider.setValue(int(image.shape[0] / 2))
        else:
            self.update_image_view()
        if image_update:
            pyplot.xlim(ax_lim)
            pyplot.ylim(ay_lim)

    def update_segmentation_image(self):
        if not self.segment.segmentation_changed:
            return
        self.segmentation = np.copy(self.segment.get_segmentation())
        self.segmentation[self.segmentation > 0] += 2
        self.rgb_segmentation = label_to_rgb(self.segmentation)

    def get_rgb_segmentation_and_mask(self):
        if self.rgb_segmentation.ndim == 2:
            return self.rgb_segmentation, self.segment.get_segmentation()
        else:
            return self.rgb_segmentation[self.layer_num], self.segment.get_segmentation()[self.layer_num]


class CropSet(QDialog):
    def __init__(self, max_size, current_size):
        super(CropSet, self).__init__()
        self.max_size = max_size
        self.min_x = QDoubleSpinBox(self)
        self.max_x = QDoubleSpinBox(self)
        self.min_y = QDoubleSpinBox(self)
        self.max_y = QDoubleSpinBox(self)
        self.min_x.setDecimals(3)
        self.max_x.setDecimals(3)
        self.min_y.setDecimals(3)
        self.max_y.setDecimals(3)
        self.min_x.setRange(0, max_size[0])
        self.min_x.setValue(current_size[0][0])
        self.max_x.setRange(0, max_size[0])
        self.max_x.setValue(current_size[0][1])
        self.min_y.setRange(0, max_size[1])
        self.min_y.setValue(current_size[1][0])
        self.max_y.setRange(0, max_size[1])
        self.max_y.setValue(current_size[1][1])
        self.min_x.valueChanged[float].connect(self.min_x_changed)
        self.max_x.valueChanged[float].connect(self.max_x_changed)
        self.min_y.valueChanged[float].connect(self.min_y_changed)
        self.max_y.valueChanged[float].connect(self.max_y_changed)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Image size: {}".format(max_size)))
        range_layout = QGridLayout()
        range_layout.addWidget(QLabel("Width:"), 0, 0)
        x_layout = QHBoxLayout()
        x_layout.addWidget(self.min_x)
        x_layout.addWidget(self.max_x)
        range_layout.addLayout(x_layout, 0, 1)
        range_layout.addWidget(QLabel("Height"), 1, 0)
        y_layout = QHBoxLayout()
        y_layout.addWidget(self.min_y)
        y_layout.addWidget(self.max_y)
        range_layout.addLayout(y_layout, 1, 1)
        layout.addLayout(range_layout)
        button_layout = QHBoxLayout()
        close_button = QPushButton("Cancel")
        close_button.clicked.connect(self.close)
        button_layout.addWidget(close_button)
        button_layout.addStretch()
        save_button = QPushButton("Crop image")
        save_button.clicked.connect(self.accept)
        button_layout.addWidget(save_button)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def min_x_changed(self, val):
        self.max_x.setMinimum(val + 1)

    def max_x_changed(self, val):
        self.min_x.setMaximum(val - 1)

    def min_y_changed(self, val):
        self.max_y.setMinimum(val + 1)

    def max_y_changed(self, val):
        self.min_y.setMaximum(val - 1)

    def get_range(self):
        return (self.min_x.value(), self.max_x.value()), (self.min_y.value(), self.max_y.value())


class DrawObject(object):
    """
    :type settings: Settings
    """
    def __init__(self, settings, segment, update_fun):
        """
        :type settings: Settings
        :type segment: Segment
        :param update_fun:
        """
        self.settings = settings
        self.segment = segment
        self.mouse_down = False
        self.draw_on = False
        self.draw_canvas = None
        self.prev_x = None
        self.prev_y = None
        self.original_rgb_image = None
        self.rgb_image = None
        self.labeled_rgb_image = None
        self.layer_num = 0
        im = [np.arange(5)]
        rgb_im = sitk.GetArrayFromImage(sitk.LabelToRGB(sitk.GetImageFromArray(im)))
        self.draw_colors = rgb_im[0]
        self.update_fun = update_fun
        self.draw_value = 0
        self.draw_fun = None
        self.value = 1
        self.click_history = []
        self.history = []
        self.f_x = 0
        self.f_y = 0

    def set_layer_num(self, layer_num):
        self.layer_num = layer_num

    def generate_set_mode(self, t):
        def set_mode():
            self.draw_on = True
            self.value = t.value
        return set_mode

    def set_draw_mode(self):
        self.draw_on = True
        self.value = 1

    def set_erase_mode(self):
        self.draw_on = True
        self.value = 2

    def set_image(self, image):
        self.draw_canvas = np.zeros(image.shape, dtype=np.uint8)
        self.segment.draw_canvas = self.draw_canvas

    def draw(self, pos):
        if len(self.original_rgb_image.shape) == 3:
            pos = pos[1:]
        self.click_history.append((pos, self.draw_canvas[pos]))
        self.draw_canvas[pos] = self.value
        self.rgb_image[pos] = self.original_rgb_image[pos] * (1 - self.settings.overlay) + \
                              self.draw_colors[self.value] * self.settings.overlay

    def erase(self, pos):
        if len(self.original_rgb_image.shape) == 3:
            pos = pos[1:]
        self.click_history.append((pos, self.draw_canvas[pos]))
        self.draw_canvas[pos] = 0
        self.rgb_image[pos] = self.labeled_rgb_image[pos]

    def clean(self):
        self.draw_canvas[...] = 0
        self.rgb_image[...] = self.labeled_rgb_image[...]
        self.segment.draw_counter = 0
        self.update_fun()
        self.settings.metadata_changed()

    def on_mouse_down(self, event):
        self.mouse_down = True
        self.click_history = []
        if self.draw_on and event.xdata is not None and event.ydata is not None:
            ix, iy = int(event.xdata + 0.5), int(event.ydata + 0.5)
            if len(self.original_rgb_image.shape) > 3:
                val = self.draw_canvas[self.layer_num, iy, ix]
            else:
                val = self.draw_canvas[iy, ix]
            if val in [0, self.value]:
                self.draw_fun = self.draw
            else:
                self.draw_fun = self.erase
            self.draw_fun((self.layer_num, iy, ix))
            self.f_x = event.xdata + 0.5
            self.f_y = event.ydata + 0.5
            self.update_fun()

    def on_mouse_move(self, event):
        if self.value > 2:
            return
        if self.mouse_down and self.draw_on and event.xdata is not None and event.ydata is not None:
            f_x, f_y = event.xdata + 0.5, event.ydata + 0.5
            # ix, iy = int(f_x), int(f_y)
            max_dist = max(abs(f_x - self.f_x), abs(f_y - self.f_y))
            rep_num = int(max_dist * 2) + 2
            points = set()
            for fx, fy in zip(np.linspace(f_x, self.f_x, num=rep_num), np.linspace(f_y, self.f_y, num=rep_num)):
                points.add((int(fx), int(fy)))
            points.remove((int(self.f_x), int(self.f_y)))
            for fx, fy in points:
                self.draw_fun((self.layer_num, fy, fx))
            self.f_x = f_x
            self.f_y = f_y
            self.update_fun()
        elif event.xdata is not None and event.ydata is not None:
            self.f_x, self.f_y = int(event.xdata) + 0.5, (event.ydata + 0.5)
            self.update_fun(False)

    def on_mouse_up(self, _):
        self.history.append(self.click_history)
        self.segment.draw_counter = np.count_nonzero(self.draw_canvas)
        self.mouse_down = False
        self.settings.metadata_changed()
