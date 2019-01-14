import os
from PyQt5.Qt import QMainWindow, QFont, QApplication, QIcon
from PyQt5.QtWidgets import QCheckBox, QLabel, QTextEdit, QStatusBar, QFileDialog, QMessageBox, QWidget, QHBoxLayout,\
    QVBoxLayout
import tifffile
import numpy as np
import logging

from partseg_utils.global_settings import config_folder, big_font_size, static_file_folder
from common_gui.universal_gui_part import set_position
from .backend import Settings, Segment
from .gui import MainMenu, InfoMenu, ColormapCanvas, SynchronizeSliders, HelpWindow, Credits, ImageExporter, synchronize_zoom
from .image_view import MyCanvas, MyDrawCanvas
from .batch_window import BatchWindow
from .io_functions import load_project


class MainWindow(QMainWindow):
    def __init__(self, title, path_to_open):
        super(MainWindow, self).__init__()
        self.open_path = path_to_open
        self.setWindowTitle(title)
        self.title = title
        self.settings = Settings(os.path.join(config_folder, "settings.json"))
        self.segment = Segment(self.settings)
        self.main_menu = MainMenu(self.settings, self.segment, self)
        self.info_menu = InfoMenu(self.settings, self.segment, self)

        self.normal_image_canvas = MyCanvas((12, 12), self.settings, self.info_menu, self)
        self.colormap_image_canvas = ColormapCanvas((1, 12), self.settings, self)
        self.segmented_image_canvas = MyDrawCanvas((12, 12), self.settings, self.info_menu, self.segment, self)
        self.segmented_image_canvas.segment.add_segmentation_callback((self.update_object_information,))
        self.normal_image_canvas.update_elements_positions()
        self.segmented_image_canvas.update_elements_positions()
        self.slider_swap = QCheckBox("Synchronize\nsliders", self)
        self.slider_swap.setChecked(True)
        self.sync = SynchronizeSliders(self.normal_image_canvas.slider, self.segmented_image_canvas.slider,
                                       self.slider_swap)
        self.colormap_image_canvas.set_bottom_widget(self.slider_swap)
        self.zoom_sync = QCheckBox("Synchronize\nzoom", self)
        # self.zoom_sync.setDisabled(True)
        synchronize_zoom(self.normal_image_canvas, self.segmented_image_canvas, self.zoom_sync)
        self.colormap_image_canvas.set_top_widget(self.zoom_sync)
        self.colormap_image_canvas.set_layout()

        # noinspection PyArgumentList
        big_font = QFont(QApplication.font())
        big_font.setPointSize(big_font_size)

        self.object_count = QLabel(self)
        self.object_count.setFont(big_font)
        self.object_count.setFixedWidth(150)
        self.object_size_list = QTextEdit(self)
        self.object_size_list.setReadOnly(True)
        self.object_size_list.setFont(big_font)
        self.object_size_list.setMinimumWidth(500)
        self.object_size_list.setMaximumHeight(30)
        # self.object_size_list.setTextInteractionFlags(Qt.TextSelectableByMouse)
        # self.object_size_list_area.setWidget(self.object_size_list)
        # self.object_size_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.object_size_list.setMinimumHeight(200)
        # self.object_size_list.setWordWrap(True)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.settings.add_image_callback((self.set_info, str))

        # self.setGeometry(0, 0,  1400, 720)
        icon = QIcon(os.path.join(static_file_folder, 'icons', "icon.png"))
        self.setWindowIcon(icon)
        menu_bar = self.menuBar()
        menu = menu_bar.addMenu("File")

        menu.addAction("Load").triggered.connect(self.main_menu.open_file)
        save = menu.addAction("Save")
        save.setDisabled(True)
        save.triggered.connect(self.main_menu.save_file)
        export = menu.addAction("Export")
        export.setDisabled(True)
        export.triggered.connect(self.export)
        self.main_menu.enable_list.extend([save, export])
        batch = menu.addAction("Batch processing")
        batch.triggered.connect(self.batch_view)
        exit_menu = menu.addAction("Exit")
        exit_menu.triggered.connect(self.close)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("Help").triggered.connect(self.help)
        help_menu.addAction("Credits").triggered.connect(self.credits)
        self.credits_widget = None
        self.help_widget = None
        self.batch_widget = None

        self.update_objects_positions()
        self.settings.add_image(tifffile.imread(os.path.join(static_file_folder, 'initial_images', "clean_segment.tiff")), "")

    def batch_view(self):
        if self.batch_widget is not None and self.batch_widget.isVisible():
            self.batch_widget.activateWindow()
            return
        self.batch_widget = BatchWindow(self.settings)
        self.batch_widget.show()

    def help(self):
        self.help_widget = HelpWindow()
        self.help_widget.show()

    def credits(self):
        self.credits_widget = Credits(self)
        self.credits_widget.exec_()

    def set_info(self, image, txt):
        self.statusBar.showMessage("{} {}".format(txt, image.shape))
        self.setWindowTitle("{}: {} {}".format(self.title, os.path.basename(txt), image.shape))

    def export(self):
        dial = QFileDialog(self, "Save data")
        if self.settings.export_directory is not None:
            dial.setDirectory(self.settings.export_directory)
        dial.setFileMode(QFileDialog.AnyFile)
        filters = ["Labeled layer (*.png)", "Clean layer (*.png)", "Only label (*.png)"]
        dial.setAcceptMode(QFileDialog.AcceptSave)
        dial.setNameFilters(filters)
        default_name = os.path.splitext(os.path.basename(self.settings.file_path))[0]
        dial.selectFile(default_name)
        if self.settings.export_filter is not None:
            dial.selectNameFilter(self.settings.export_filter)
        if dial.exec_():
            file_path = str(dial.selectedFiles()[0])
            if os.path.splitext(file_path)[1] != ".png":
                file_path += ".png"
                if os.path.exists(file_path):
                    # noinspection PyCallByClass
                    ret = QMessageBox.warning(self, "File exist", os.path.basename(file_path) +
                                              " already exists.\nDo you want to replace it?",
                                              QMessageBox.No, QMessageBox.Yes)
                    if ret == QMessageBox.No:
                        self.export()
                        return
            selected_filter = str(dial.selectedNameFilter())
            self.settings.export_filter = selected_filter
            self.settings.export_directory = os.path.dirname(file_path)
            if selected_filter == "Labeled layer (*.png)":
                ie = ImageExporter(self.segmented_image_canvas, file_path, selected_filter, self)
                ie.exec_()
            elif selected_filter == "Clean layer (*.png)":
                ie = ImageExporter(self.normal_image_canvas, file_path, selected_filter, self)
                ie.exec_()
            elif selected_filter == "Only label (*.png)":
                seg, mask = self.segmented_image_canvas.get_rgb_segmentation_and_mask()
                seg = np.dstack((seg, np.zeros(seg.shape[:-1], dtype=np.uint8)))
                seg[..., 3][mask > 0] = 255
                ie = ImageExporter(self.segmented_image_canvas, file_path, selected_filter, self,
                                   image=seg)
                ie.exec_()
            else:
                _ = QMessageBox.critical(self, "Save error", "Option unknown")

    def showEvent(self, _):
        try:
            if self.open_path is not None:
                if os.path.splitext(self.open_path)[1] in ['.bz2', ".tbz2", ".gz", "tgz"]:
                    load_project(self.open_path, self.settings, self.segment)
                elif os.path.splitext(self.open_path)[1] in ['.tif', '.tiff', '*.lsm']:
                    im = tifffile.imread(self.open_path)
                    if im.ndim < 4:
                        self.settings.add_image(im, self.open_path)
                    else:
                        return
                for el in self.main_menu.enable_list:
                    el.setEnabled(True)
                self.main_menu.settings_changed()
        except Exception as e:
            logging.warning(e.message)

    def update_objects_positions(self):
        widget = QWidget()
        main_layout = QVBoxLayout()
        menu_layout = QHBoxLayout()
        menu_layout.addWidget(self.main_menu)
        menu_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(menu_layout)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.info_menu)
        info_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addLayout(info_layout)
        image_layout = QHBoxLayout()
        image_layout.setSpacing(0)
        image_layout.addWidget(self.colormap_image_canvas)
        image_layout.addWidget(self.normal_image_canvas)
        image_layout.addWidget(self.segmented_image_canvas)
        main_layout.addLayout(image_layout)
        main_layout.addSpacing(5)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.object_count)
        info_layout.addWidget(self.object_size_list)
        main_layout.addLayout(info_layout)
        main_layout.addStretch()
        main_layout.setSpacing(0)
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def resizeEvent(self, *args, **kwargs):
        super(MainWindow, self).resizeEvent(*args, **kwargs)

    def update_objects_positions2(self):
        self.normal_image_canvas.move(10, 40)
        # noinspection PyTypeChecker
        set_position(self.colormap_image_canvas, self.normal_image_canvas, 0)
        # noinspection PyTypeChecker
        set_position(self.segmented_image_canvas, self.colormap_image_canvas, 0)
        col_pos = self.colormap_image_canvas.pos()
        self.slider_swap.move(col_pos.x() + 5,
                              col_pos.y() + self.colormap_image_canvas.height() - 35)
        # self.infoText.move()

        norm_pos = self.normal_image_canvas.pos()
        self.object_count.move(norm_pos.x(),
                               norm_pos.y() + self.normal_image_canvas.height() + 20)
        self.object_size_list.move(self.object_count.pos().x() + 150, self.object_count.pos().y())

    def update_object_information(self, info_array):
        """:type info_array: np.ndarray"""
        self.object_count.setText("Object num: {0}".format(str(info_array.size)))
        self.object_size_list.setText("Objects size: {}".format(list(info_array)))

    def closeEvent(self, event):
        logging.debug("Close: {}".format(os.path.join(config_folder, "settings.json")))
        if self.batch_widget is not None and self.batch_widget.isVisible():
            if self.batch_widget.is_working():
                ret = QMessageBox.warning(self, "Batch work", "Batch work is not finished. "
                                                              "Would you like to terminate it?",
                                          QMessageBox.No | QMessageBox.Yes)
                if ret == QMessageBox.Yes:
                    self.batch_widget.terminate()
                    self.batch_widget.close()
                else:
                    event.ignore()
                    return
            else:
                self.batch_widget.close()
        self.settings.dump(os.path.join(config_folder, "settings.json"))
        if self.main_menu.advanced_window is not None and self.main_menu.advanced_window.isVisible():
            self.main_menu.advanced_window.close()
