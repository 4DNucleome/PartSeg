from __future__ import division

import os

import appdirs
import numpy as np
import tifffile as tif
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QFileDialog, QMessageBox, QVBoxLayout, QCheckBox, \
    QComboBox, QDoubleSpinBox, QSpinBox, QStackedLayout, QProgressBar, QLabel, QAbstractSpinBox, QFormLayout, \
    QTabWidget, QMainWindow, QSizePolicy

from common_gui.channel_control import ChannelControl
from common_gui.universal_gui_part import right_label

from common_gui.flow_layout import FlowLayout
from .image_view import StackImageView
from partseg.batch_window import AddFiles
from partseg.io_functions import load_stack_segmentation
from project_utils.global_settings import static_file_folder
from project_utils.universal_const import UNITS_LIST
from project_utils.utils import SynchronizeValues
from stackseg.stack_algorithm.algorithm_description import stack_algorithm_dict, AlgorithmSettingsWidget, BatchProceed
from stackseg.stack_settings import StackSettings

app_name = "StackSeg"
app_lab = "LFSG"
config_folder = appdirs.user_data_dir(app_name, app_lab)

class MainMenu(QWidget):
    image_loaded = pyqtSignal()

    def __init__(self, settings):
        """
        :type settings: StackSettings
        :param settings:
        """
        super(MainMenu, self).__init__()
        self.settings = settings
        self.load_image_btn = QPushButton("Load image (also from mask)")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_segmentation_btn = QPushButton("Load segmentation")
        self.load_segmentation_btn.clicked.connect(self.load_segmentation)
        self.save_segmentation_btn = QPushButton("Save segmentation")
        self.save_segmentation_btn.clicked.connect(self.save_segmentation)
        self.save_catted_parts = QPushButton("Save components")
        self.save_catted_parts.clicked.connect(self.save_result)
        layout = QHBoxLayout()
        layout.addWidget(self.load_image_btn)
        layout.addWidget(self.load_segmentation_btn)
        layout.addWidget(self.save_catted_parts)
        layout.addWidget(self.save_segmentation_btn)
        self.setLayout(layout)

    def load_image(self):
        try:
            dial = QFileDialog()
            dial.setFileMode(QFileDialog.ExistingFile)
            dial.setDirectory(self.settings.get("io.load_image_directory", ""))
            filters = ["raw image (*.tiff *.tif *.lsm)", "image from mask (*.seg *.tgz)"]
            dial.setNameFilters(filters)
            dial.selectNameFilter(self.settings.get("io.load_image_filter", filters[0]))
            if not dial.exec_():
                return
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.load_image_directory", os.path.dirname(str(file_path)))
            self.settings.set("io.load_image_filter", dial.selectedNameFilter())
            if dial.selectedNameFilter() == filters[1]:
                segmentation, metadata = load_stack_segmentation(file_path)
                if "base_file" not in metadata:
                    QMessageBox.warning(self, "Open error", "No information about base file")
                if not os.path.exists(metadata["base_file"]):
                    QMessageBox.warning(self, "Open error", "Base file not found")
                im = tif.imread(metadata["base_file"])
                self.settings.image = im, metadata["base_file"]
                self.settings.set_segmentation(segmentation, metadata)
            else:
                im = tif.imread(file_path)
                self.settings.image = im, file_path
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "Open error", "Exception occurred {}".format(e))
        # self.image_loaded.emit()

    def load_segmentation(self):
        try:

            dial = QFileDialog()
            dial.setFileMode(QFileDialog.ExistingFile)
            dial.setDirectory(self.settings.get("io.open_segmentation_directory", ""))
            filters = ["segmentation (*.seg *.tgz)"]
            dial.setNameFilters(filters)
            if not dial.exec_():
                return
            file_path = str(dial.selectedFiles()[0])
            self.settings.set("io.open_segmentation_directory", os.path.dirname(str(file_path)))
            self.settings.load_segmentation(file_path)
        except Exception as e:
            QMessageBox.warning(self, "Open error", "Exception occurred {}".format(e))

    def save_segmentation(self):
        if self.settings.segmentation is None:
            QMessageBox.warning(self, "No segmentation", "No segmentation to save")
            return
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setDirectory(self.settings.get("io.save_segmentation_directory", ""))
        dial.setAcceptMode(QFileDialog.AcceptSave)
        filters = ["segmentation (*.seg *.tgz)"]
        dial.setNameFilters(filters)
        if not dial.exec_():
            return
        file_path = str(dial.selectedFiles()[0])
        self.settings.set("io.save_segmentation_directory", os.path.dirname(str(file_path)))
        #self.settings.save_directory = os.path.dirname(str(file_path))
        self.settings.save_segmentation(file_path)

    def save_result(self):
        if self.settings.segmentation is None:
            QMessageBox.warning(self, "No components", "No components to save")
            return
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.Directory)
        dial.setDirectory(self.settings.get("io.save_components_directory", ""))
        if not dial.exec_():
            return
        file_path = str(dial.selectedFiles()[0])
        self.settings.set("io.save_components_directory", os.path.dirname(str(file_path)))
        self.settings.save_result(file_path)


class ChosenComponents(QWidget):
    """
    :type check_box: dict[int, QCheckBox]
    """
    check_change_signal = pyqtSignal()
    def __init__(self):
        super(ChosenComponents, self).__init__()
        # self.setLayout(FlowLayout())
        self.check_box = dict()
        self.check_all_btn = QPushButton("Check all")
        self.check_all_btn.clicked.connect(self.check_all)
        self.un_check_all_btn = QPushButton("Un check all")
        self.un_check_all_btn.clicked.connect(self.un_check_all)
        main_layout = QVBoxLayout()
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.check_all_btn)
        btn_layout.addWidget(self.un_check_all_btn)
        self.check_layout = FlowLayout()
        main_layout.addLayout(btn_layout)
        main_layout.addLayout(self.check_layout)
        self.setLayout(main_layout)

    def other_component_choose(self, num):
        check = self.check_box[num]
        check.setChecked(not check.isChecked())

    def check_all(self):
        for el in self.check_box.values():
            el.setChecked(True)

    def un_check_all(self):
        for el in self.check_box.values():
            el.setChecked(False)

    def set_chose(self, components_index, chosen_components):
        self.blockSignals(True)
        self.check_layout.clear()
        for el in self.check_box.values():
            """:type el: QCheckBox"""
            el.deleteLater()
            el.stateChanged.disconnect()
        self.check_box.clear()
        chosen_components = set(chosen_components)
        for el in components_index:
            check = QCheckBox(str(el))
            if el in chosen_components:
                check.setChecked(True)
            check.stateChanged.connect(self.check_change)
            self.check_box[el] = check
            self.check_layout.addWidget(check)
        self.blockSignals(False)
        self.update()
        self.check_change_signal.emit()


    def check_change(self):
        self.check_change_signal.emit()

    def change_state(self, num, val):
        self.check_box[num].setChecked(val)

    def get_state(self, num: int) -> bool:
        return self.check_box[num].isChecked()

    def get_chosen(self):
        res = []
        for num, check in self.check_box.items():
            if check.isChecked():
                res.append(num)
        return res

    def get_mask(self):
        res = [0]
        for _, check in sorted(self.check_box.items()):
            res.append(check.isChecked())
        return np.array(res, dtype=np.uint8)


class AlgorithmOptions(QWidget):
    def __init__(self, settings, control_view, component_checker):
        """
        :type control_view: ImageState
        :type settings: ImageSettings
        :param settings:
        :param control_view:
        """
        super(AlgorithmOptions, self).__init__()
        self.settings = settings
        self.algorithm_choose = QComboBox()
        self.show_result = QComboBox() #  QCheckBox("Show result")
        self.show_result.addItems(["Not show", "Show results", "Show choosen"])
        self.show_result.setCurrentIndex(control_view.show_label)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.1)
        self.opacity.setValue(control_view.opacity)
        self.only_borders = QCheckBox("Only borders")
        self.only_borders.setChecked(control_view.only_borders)
        self.borders_thick = QSpinBox()
        self.borders_thick.setRange(1, 11)
        self.borders_thick.setSingleStep(2)
        self.borders_thick.setValue(control_view.borders_thick)
        self.borders_thick.valueChanged.connect(self.border_value_check)
        self.execute_btn = QPushButton("Execute")
        self.execute_all_btn = QPushButton("Execute all")
        self.execute_all_btn.setDisabled(True)
        self.block_execute_all_btn = False
        self.stack_layout = QStackedLayout()
        self.choose_components = ChosenComponents()
        self.choose_components.check_change_signal.connect(control_view.components_change)
        widgets_list = []
        for name, val in stack_algorithm_dict.items():
            self.algorithm_choose.addItem(name)
            widget = AlgorithmSettingsWidget(settings, name, *val)
            widgets_list.append(widget)
            widget.algorithm.execution_done.connect(self.execution_done)
            widget.algorithm.progress_signal.connect(self.progress_info)
            self.stack_layout.addWidget(widget)
        # WARNING works only with one channels algorithms
        SynchronizeValues.add_synchronization("channels_chose", widgets_list)
        self.chosen_list = []
        self.progress_bar = QProgressBar()
        self.progress_bar.setHidden(True)
        self.progress_info_lab = QLabel()
        self.progress_info_lab.setHidden(True)
        self.file_list = []
        self.batch_process = BatchProceed()
        self.batch_process.progress_signal.connect(self.progress_info)
        self.batch_process.error_signal.connect(self.execution_all_error)
        self.batch_process.execution_done.connect(self.execution_all_done)
        self.is_batch_process = False

        main_layout = QVBoxLayout()
        opt_layout = QHBoxLayout()
        opt_layout.addWidget(self.show_result)
        opt_layout.addWidget(right_label("Opacity:"))
        opt_layout.addWidget(self.opacity)
        main_layout.addLayout(opt_layout)
        opt_layout2 = QHBoxLayout()
        opt_layout2.addWidget(self.only_borders)
        opt_layout2.addWidget(right_label("Border thick:"))
        opt_layout2.addWidget(self.borders_thick)
        main_layout.addLayout(opt_layout2)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.execute_btn)
        btn_layout.addWidget(self.execute_all_btn)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_info_lab)
        main_layout.addWidget(self.algorithm_choose)
        main_layout.addLayout(self.stack_layout)
        main_layout.addWidget(self.choose_components)
        main_layout.addStretch()
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)
        self.setLayout(main_layout)

        self.algorithm_choose.currentIndexChanged.connect(self.stack_layout.setCurrentIndex)
        self.execute_btn.clicked.connect(self.execute_action)
        self.execute_all_btn.clicked.connect(self.execute_all_action)
        self.opacity.valueChanged.connect(control_view.set_opacity)
        self.show_result.currentIndexChanged.connect(control_view.set_show_label)
        self.only_borders.stateChanged.connect(control_view.set_borders)
        self.borders_thick.valueChanged.connect(control_view.set_borders_thick)
        settings.image_changed.connect(self.image_changed)
        component_checker.component_clicked.connect(self.choose_components.other_component_choose)
        settings.chosen_components_widget = self.choose_components

    def border_value_check(self, value):
        if value % 2 == 0:
            self.borders_thick.setValue(value+1)

    def file_list_change(self, val):
        print("FF:", val)
        self.file_list = val
        if len(self.file_list) > 0 and not self.block_execute_all_btn:
            self.execute_all_btn.setEnabled(True)
        else:
            self.execute_all_btn.setDisabled(True)

    def get_chosen_components(self):
        return sorted(self.choose_components.get_chosen())

    @property
    def segmentation(self):
        return self.settings.segmentation

    @segmentation.setter
    def segmentation(self, val):
        self.settings.segmentation = val

    def image_changed(self):
        self.segmentation = None
        self.choose_components.set_chose([], [])

    def execute_all_action(self):
        dial = QFileDialog()
        dial.setFileMode(QFileDialog.Directory)
        dial.setDirectory(self.settings.get("io.save_batch", self.settings.get("io.save_segmentation_directory", "")))
        if not dial.exec_():
            return
        folder_path = str(dial.selectedFiles()[0])
        self.settings.set("io.save_batch", folder_path)
        self.execute_all_btn.setDisabled(True)
        self.block_execute_all_btn = True
        self.is_batch_process = True
        self.execute_btn.setDisabled(True)
        self.progress_bar.setHidden(False)
        self.progress_bar.setRange(0, len(self.file_list))
        self.progress_bar.setValue(0)

        widget = self.stack_layout.currentWidget()
        parameters = widget.get_values()
        self.batch_process.set_parameters(type(widget.algorithm), parameters, widget.channel_num(),
                                          self.file_list, folder_path)
        self.batch_process.start()

    def execution_all_error(self, text):
        QMessageBox.warning(self, "Proceed error", text)

    def execution_all_done(self):
        self.execute_btn.setEnabled(True)
        self.block_execute_all_btn = False
        if len(self.file_list) > 0:
            self.execute_all_btn.setEnabled(True)
        self.progress_bar.setHidden(True)
        self.progress_info_lab.setHidden(True)

    def execute_action(self):
        self.execute_btn.setDisabled(True)
        self.execute_all_btn.setDisabled(True)
        self.block_execute_all_btn = True
        self.is_batch_process = False
        self.progress_bar.setRange(0, 0)
        chosen = sorted(self.choose_components.get_chosen())
        if len(chosen) == 0:
            blank = None
        else:
            if len(chosen) > 250:
                blank = np.zeros(self.segmentation.shape, dtype=np.uint16)
            else:
                blank = np.zeros(self.segmentation.shape, dtype=np.uint8)
            for i, v in enumerate(chosen):
                blank[self.segmentation == v] = i + 1
        self.progress_bar.setHidden(False)
        widget = self.stack_layout.currentWidget()
        widget.execute(blank)
        self.chosen_list = chosen

    def progress_info(self, text, num):
        self.progress_info_lab.setVisible(True)
        self.progress_info_lab.setText(text)
        if self.is_batch_process:
            self.progress_bar.setValue(num)

    def execution_done(self, segmentation):
        self.segmentation = segmentation
        self.choose_components.set_chose(range(1, segmentation.max() + 1), np.arange(len(self.chosen_list)) + 1)
        self.execute_btn.setEnabled(True)
        self.block_execute_all_btn = False
        if len(self.file_list) > 0:
            self.execute_all_btn.setEnabled(True)
        self.progress_bar.setHidden(True)
        self.progress_info_lab.setHidden(True)


class ImageInformation(QWidget):
    def __init__(self, settings, parent=None):
        """:type settings: ImageSettings"""
        super(ImageInformation, self).__init__(parent)
        self._settings = settings
        self.path = QLabel("<b>Path:</b> example image")
        self.path.setWordWrap(True)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        for i, el in enumerate(self.spacing):
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 1000)
            el.setValue(self._settings.image_spacing[i])
            el.valueChanged.connect(self.image_spacing_change)
        self.units = QComboBox()
        self.units.addItems(UNITS_LIST)
        self.units.setCurrentIndex(2)

        self.add_files = AddFiles(settings, btn_layout=FlowLayout)

        spacing_layout = QFormLayout()
        spacing_layout.addRow("x:", self.spacing[0])
        spacing_layout.addRow("y:", self.spacing[1])
        spacing_layout.addRow("z:", self.spacing[2])
        spacing_layout.addRow("Units:", self.units)

        layout = QVBoxLayout()
        layout.addWidget(self.path)
        layout.addWidget(QLabel("Image spacing:"))
        layout.addLayout(spacing_layout)
        layout.addWidget(self.add_files)
        layout.addStretch()
        self.setLayout(layout)
        self._settings.image_changed[str].connect(self.set_image_path)

    def set_image_path(self, value):
        self.path.setText("<b>Path:</b> {}".format(value))

    def image_spacing_change(self):
        self._settings.image_spacing = [el.value() for el in self.spacing]


class Options(QTabWidget):
    def __init__(self, settings, control_view, component_checker, parent=None):
        super(Options, self).__init__(parent)
        self._settings = settings
        self.algorithm_options = AlgorithmOptions(settings, control_view, component_checker)
        self.image_properties = ImageInformation(settings, parent)
        self.image_properties.add_files.file_list_changed.connect(self.algorithm_options.file_list_change)
        self.addTab(self.image_properties, "Image")
        self.addTab(self.algorithm_options, "Segmentation")

    def get_chosen_components(self):
        return self.algorithm_options.get_chosen_components()


class MainWindow(QMainWindow):
    def __init__(self, title):
        super(MainWindow, self).__init__()
        self.setWindowTitle(title)
        self.settings = StackSettings()
        if os.path.exists(os.path.join(config_folder, "settings.json")):
            self.settings.load(os.path.join(config_folder, "settings.json"))
        self.main_menu = MainMenu(self.settings)
        self.channel_control = ChannelControl(self.settings)
        self.image_view = StackImageView(self.settings, self.channel_control)
        self.image_view.setMinimumWidth(450)
        self.info_text = QLabel()
        self.info_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.image_view.text_info_change.connect(self.info_text.setText)
        image_view_control = self.image_view.get_control_view()
        self.options_panel = Options(self.settings, image_view_control, self.image_view)
        self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)

        im = tif.imread(os.path.join(static_file_folder, 'initial_images', "stack.tif"))
        # width, height = im.shape
        # im = colors.PowerNorm(gamma=1, vmin=im.min(), vmax=im.max())(im)
        # cmap = matplotlib.cm.get_cmap("cubehelix")
        # colored_image = cmap(im)
        # noinspection PyTypeChecker
        # im = np.array(colored_image * 255, dtype=np.uint8)
        # im2 = QImage(im.data, width, height, im.dtype.itemsize*width*4, QImage.Format_ARGB32)

        # self.pixmap.setPixmap(QPixmap.fromImage(im2))
        # self.im_view = pg.ImageView(self)
        # self.im_view.setImage(im)
        layout = QVBoxLayout()
        layout.addWidget(self.main_menu)
        sub_layout = QHBoxLayout()
        sub2_layout = QVBoxLayout()
        sub3_layout = QVBoxLayout()
        sub3_layout.addWidget(self.image_view, 1)
        sub3_layout.addWidget(self.info_text, 0)
        sub2_layout.addWidget(self.options_panel, 1)
        sub2_layout.addWidget(self.channel_control, 0)
        sub_layout.addLayout(sub3_layout,1)
        sub_layout.addLayout(sub2_layout,0)
        layout.addLayout(sub_layout)
        # self.pixmap.adjustSize()
        # self.pixmap.update_size(2)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        self.settings.image = im

    def image_read(self):
        print("buka1", self.settings.image.shape, self.sender())
        self.image_view.set_image(self.settings.image)

    def closeEvent(self, _):
        # print(self.settings.dump_view_profiles())
        # print(self.settings.segmentation_dict["default"].my_dict)
        self.settings.dump(os.path.join(config_folder, "settings.json"))




