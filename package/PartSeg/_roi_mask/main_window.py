import os
from functools import partial
from pathlib import Path
from typing import Type

import numpy as np
from qtpy.QtCore import QByteArray, Qt, Signal, Slot
from qtpy.QtGui import QCloseEvent, QGuiApplication, QIcon, QKeySequence, QTextOption
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import PartSegData
from PartSegCore import UNIT_SCALE, Units, state_store
from PartSegCore.io_utils import WrongFileTypeException
from PartSegCore.mask import io_functions
from PartSegCore.mask.algorithm_description import mask_algorithm_dict
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import LoadROI, LoadROIFromTIFF, LoadROIParameters, MaskProjectTuple, SaveROI
from PartSegCore.project_info import HistoryElement, HistoryProblem, calculate_mask_from_project
from PartSegImage import Image, TiffImageReader

from .._roi_mask.segmentation_info_dialog import SegmentationInfoDialog
from ..common_gui.advanced_tabs import AdvancedWindow
from ..common_gui.algorithms_description import AlgorithmChoose, AlgorithmSettingsWidget, EnumComboBox
from ..common_gui.channel_control import ChannelProperty
from ..common_gui.custom_load_dialog import CustomLoadDialog
from ..common_gui.custom_save_dialog import SaveDialog
from ..common_gui.flow_layout import FlowLayout
from ..common_gui.main_window import BaseMainMenu, BaseMainWindow
from ..common_gui.mask_widget import MaskDialogBase
from ..common_gui.multiple_file_widget import MultipleFileWidget
from ..common_gui.napari_image_view import LabelEnum
from ..common_gui.select_multiple_files import AddFiles
from ..common_gui.stack_image_view import ColorBar
from ..common_gui.universal_gui_part import right_label
from ..common_gui.waiting_dialog import ExecuteFunctionDialog
from .batch_proceed import BatchProceed, BatchTask
from .image_view import StackImageView
from .simple_measurements import SimpleMeasurements
from .stack_settings import StackSettings, get_mask

CONFIG_FOLDER = os.path.join(state_store.save_folder, "mask")


class MaskDialog(MaskDialogBase):
    def __init__(self, settings: StackSettings):
        super().__init__(settings)
        self.settings = settings

    def next_mask(self):
        project_info: MaskProjectTuple = self.settings.get_project_info()
        mask_property = self.mask_widget.get_mask_property()
        self.settings.set("mask_manager.mask_property", mask_property)
        mask = calculate_mask_from_project(mask_description=mask_property, project=project_info)

        self.settings.add_history_element(
            create_history_element_from_segmentation_tuple(
                project_info,
                mask_property,
            )
        )
        self.settings.mask = mask
        self.settings.chosen_components_widget.un_check_all()
        self.close()

    def prev_mask(self):
        history: HistoryElement = self.settings.history_pop()
        history.arrays.seek(0)
        seg = np.load(history.arrays)
        history.arrays.seek(0)
        self.settings.roi = seg["segmentation"]
        self.settings.set_segmentation(
            seg["segmentation"],
            False,
            history.roi_extraction_parameters["selected"],
            history.roi_extraction_parameters["parameters"],
        )
        self.settings.mask = seg["mask"] if "mask" in seg else None
        self.close()


class MainMenu(BaseMainMenu):
    image_loaded = Signal()

    def __init__(self, settings: StackSettings, main_window):
        """
        :type settings: StackSettings
        :param settings:
        """
        super().__init__(settings, main_window)
        self.settings = settings
        self.segmentation_cache = None
        self.read_thread = None
        self.advanced_window = None
        self.measurements_window = None
        self.load_image_btn = QPushButton("Load image")
        self.load_image_btn.clicked.connect(self.load_image)
        self.load_segmentation_btn = QPushButton("Load segmentation")
        self.load_segmentation_btn.clicked.connect(self.load_segmentation)
        self.save_segmentation_btn = QPushButton("Save segmentation")
        self.save_segmentation_btn.clicked.connect(self.save_segmentation)
        self.save_catted_parts = QPushButton("Save components")
        self.save_catted_parts.clicked.connect(self.save_result)
        self.advanced_window_btn = QPushButton("Advanced settings")
        self.advanced_window_btn.clicked.connect(self.show_advanced_window)
        self.mask_manager_btn = QPushButton("Mask manager")
        self.mask_manager_btn.clicked.connect(self.mask_manager)
        self.measurements_btn = QPushButton("Simple measurements")
        self.measurements_btn.clicked.connect(self.simple_measurement)
        self.segmentation_dialog = SegmentationInfoDialog(
            self.main_window.settings,
            self.main_window.options_panel.algorithm_options.algorithm_choose_widget.change_algorithm,
        )

        self.setContentsMargins(0, 0, 0, 0)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        # layout.setSpacing(0)
        layout.addWidget(self.load_image_btn)
        layout.addWidget(self.load_segmentation_btn)
        layout.addWidget(self.save_catted_parts)
        layout.addWidget(self.save_segmentation_btn)
        layout.addWidget(self.advanced_window_btn)
        layout.addWidget(self.mask_manager_btn)
        layout.addWidget(self.measurements_btn)
        self.setLayout(layout)

    def simple_measurement(self):
        if self.measurements_window is None:
            self.measurements_window = SimpleMeasurements(self.settings)
        self.measurements_window.show()

    def mask_manager(self):
        if self.settings.roi is None:
            QMessageBox.information(self, "No segmentation", "Cannot create mask without segmentation")
            return
        if not self.settings.chosen_components():
            QMessageBox.information(self, "No selected components", "Mask is created only from selected components")
            return
        dial = MaskDialog(self.settings)
        dial.exec_()

    def show_advanced_window(self):
        if self.advanced_window is None:
            self.advanced_window = AdvancedWindow(self.settings, ["channelcontrol"])
            # FIXME temporary workaround
            self.advanced_window.reload_list = []
        self.advanced_window.show()

    def load_image(self):
        # TODO move segmentation with image load to load_segmentaion
        dial = CustomLoadDialog(io_functions.load_dict)
        dial.setDirectory(self.settings.get("io.load_image_directory", str(Path.home())))
        default_file_path = self.settings.get("io.load_image_file", "")
        if os.path.isfile(default_file_path):
            dial.selectFile(default_file_path)
        dial.selectNameFilter(self.settings.get("io.load_data_filter", io_functions.load_dict.get_default()))
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if not dial.exec_():
            return
        load_property = dial.get_result()
        self.settings.set("io.load_image_directory", os.path.dirname(load_property.load_location[0]))
        self.settings.set("io.load_image_file", load_property.load_location[0])
        self.settings.set("io.load_data_filter", load_property.selected_filter)
        self.settings.add_load_files_history(load_property.load_location, load_property.load_class.get_name())

        def exception_hook(exception):
            if isinstance(exception, ValueError) and exception.args[0] == "not a TIFF file":
                QMessageBox.warning(self, "Open error", "Image is not proper tiff/lsm image")
            elif isinstance(exception, MemoryError):
                QMessageBox.warning(self, "Open error", "Not enough memory to read this image")
            elif isinstance(exception, IOError):
                QMessageBox.warning(self, "Open error", f"Some problem with reading from disc: {exception}")
            elif isinstance(exception, WrongFileTypeException):
                QMessageBox.warning(
                    self,
                    "Open error",
                    "No needed files inside archive. Most probably you choose file from segmentation analysis",
                )
            else:
                raise exception

        execute_dialog = ExecuteFunctionDialog(
            load_property.load_class.load,
            [load_property.load_location],
            {"metadata": {"default_spacing": self.settings.image.spacing}},
            text="Load data",
            exception_hook=exception_hook,
        )
        if execute_dialog.exec():
            result = execute_dialog.get_result()
            if result is None:
                return
            self.set_data(result)

    def set_image(self, image: Image) -> bool:
        if image is None:
            return False
        if image.is_time:
            if image.is_stack:
                QMessageBox.warning(self, "Not supported", "Data that are time data are currently not supported")
                return False

            res = QMessageBox.question(
                self,
                "Not supported",
                "Time data are currently not supported. Maybe You would like to treat time as z-stack",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if res == QMessageBox.Yes:
                image = image.swap_time_and_stack()
            else:
                return False
        self.settings.image = image
        return True

    def load_segmentation(self):
        dial = CustomLoadDialog(
            {
                LoadROI.get_name(): LoadROI,
                LoadROIParameters.get_name(): LoadROIParameters,
                LoadROIFromTIFF.get_name(): LoadROIFromTIFF,
            }
        )
        dial.setDirectory(self.settings.get("io.open_segmentation_directory", str(Path.home())))
        dial.setHistory(dial.history() + self.settings.get_path_history())
        if not dial.exec_():
            return
        load_property = dial.get_result()
        self.settings.set("io.open_segmentation_directory", os.path.dirname(load_property.load_location[0]))
        self.settings.add_path_history(os.path.dirname(load_property.load_location[0]))

        def exception_hook(exception):
            mess = QMessageBox(self)
            if isinstance(exception, ValueError) and exception.args[0] == "Segmentation do not fit to image":
                mess.warning(self, "Open error", "Segmentation do not fit to image")
            elif isinstance(exception, MemoryError):
                mess.warning(self, "Open error", "Not enough memory to read this image")
            elif isinstance(exception, IOError):
                mess.warning(self, "Open error", "Some problem with reading from disc")
            elif isinstance(exception, WrongFileTypeException):
                mess.warning(
                    self,
                    "Open error",
                    "No needed files inside archive. Most probably you choose file from segmentation analysis",
                )
            else:
                raise exception

        dial = ExecuteFunctionDialog(
            load_property.load_class.load,
            [load_property.load_location],
            text="Load segmentation",
            exception_hook=exception_hook,
        )
        if dial.exec():
            result = dial.get_result()
            if result is None:
                QMessageBox.critical(self, "Data Load fail", "Fail of loading data")
                return
            if result.roi is not None:
                try:
                    self.settings.set_project_info(dial.get_result())
                    return
                except ValueError as e:
                    if e.args != ("Segmentation do not fit to image",):
                        raise
                    self.segmentation_dialog.set_additional_text(
                        "Segmentation do not fit to image, maybe you would lie to load parameters only."
                    )
                except HistoryProblem:
                    QMessageBox().warning(
                        self,
                        "Load Problem",
                        "You set to save selected components when loading "
                        "another segmentation but history is incomatybile",
                    )

            else:
                self.segmentation_dialog.set_additional_text("")
            self.segmentation_dialog.set_parameters_dict(result.roi_extraction_parameters)
            self.segmentation_dialog.show()

    def save_segmentation(self):
        if self.settings.roi is None:
            QMessageBox.warning(self, "No segmentation", "No segmentation to save")
            return
        dial = SaveDialog(io_functions.save_segmentation_dict, False, history=self.settings.get_path_history())
        dial.setDirectory(self.settings.get("io.save_segmentation_directory", str(Path.home())))
        dial.selectFile(os.path.splitext(os.path.basename(self.settings.image_path))[0] + ".seg")
        if not dial.exec_():
            return
        save_location, _selected_filter, save_class, values = dial.get_result()
        self.settings.set("io.save_segmentation_directory", os.path.dirname(str(save_location)))
        self.settings.add_path_history(os.path.dirname(str(save_location)))
        # self.settings.save_directory = os.path.dirname(str(file_path))

        def exception_hook(exception):
            QMessageBox.critical(self, "Save error", f"Error on disc operation. Text: {exception}", QMessageBox.Ok)
            raise exception

        dial = ExecuteFunctionDialog(
            save_class.save,
            [save_location, self.settings.get_project_info(), values],
            text="Save segmentation",
            exception_hook=exception_hook,
        )
        dial.exec()

    def save_result(self):
        if self.settings.image_path is not None and QMessageBox.Yes == QMessageBox.question(
            self, "Copy", "Copy name to clipboard?", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
        ):
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(os.path.splitext(os.path.basename(self.settings.image_path))[0])

        if self.settings.roi is None or len(self.settings.sizes) == 1:
            QMessageBox.warning(self, "No components", "No components to save")
            return
        dial = SaveDialog(
            io_functions.save_components_dict,
            False,
            history=self.settings.get_path_history(),
            file_mode=QFileDialog.Directory,
        )
        dial.setDirectory(self.settings.get("io.save_components_directory", str(Path.home())))
        dial.selectFile(os.path.splitext(os.path.basename(self.settings.image_path))[0])
        if not dial.exec_():
            return
        res = dial.get_result()
        potential_names = self.settings.get_file_names_for_save_result(res.save_destination)
        conflict = []
        for el in potential_names:
            if os.path.exists(el):
                conflict.append(el)
        if len(conflict) > 0:
            # TODO modify because of long lists
            conflict_str = "\n".join(conflict)
            if QMessageBox.No == QMessageBox.warning(
                self,
                "Overwrite",
                f"Overwrite files:\n {conflict_str}",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ):
                self.save_result()
                return

        self.settings.set("io.save_components_directory", os.path.dirname(str(res.save_destination)))
        self.settings.add_path_history(os.path.dirname(str(res.save_destination)))

        def exception_hook(exception):
            QMessageBox.critical(self, "Save error", f"Error on disc operation. Text: {exception}", QMessageBox.Ok)

        dial = ExecuteFunctionDialog(
            res.save_class.save,
            [res.save_destination, self.settings.get_project_info(), res.parameters],
            text="Save components",
            exception_hook=exception_hook,
        )
        dial.exec()


class ComponentCheckBox(QCheckBox):
    mouse_enter = Signal(int)
    mouse_leave = Signal(int)

    def __init__(self, number: int, parent=None):
        super().__init__(str(number), parent)
        self.number = number

    def enterEvent(self, _event):
        self.mouse_enter.emit(self.number)

    def leaveEvent(self, _event):
        self.mouse_leave.emit(self.number)


class ChosenComponents(QWidget):
    """
    :type check_box: dict[int, QCheckBox]
    """

    check_change_signal = Signal()
    mouse_enter = Signal(int)
    mouse_leave = Signal(int)

    def __init__(self):
        super().__init__()
        # self.setLayout(FlowLayout())
        self.check_box = {}
        self.check_all_btn = QPushButton("Select all")
        self.check_all_btn.clicked.connect(self.check_all)
        self.un_check_all_btn = QPushButton("Unselect all")
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

    def remove_components(self):
        self.check_layout.clear()
        for el in self.check_box.values():
            el.deleteLater()
            el.stateChanged.disconnect()
            el.mouse_leave.disconnect()
            el.mouse_enter.disconnect()
        self.check_box.clear()

    def new_choose(self, num, chosen_components):
        self.set_chose(range(1, num + 1), chosen_components)

    def set_chose(self, components_index, chosen_components):
        chosen_components = set(chosen_components)
        self.blockSignals(True)
        self.remove_components()
        chosen_components = set(chosen_components)
        for el in components_index:
            check = ComponentCheckBox(el)
            if el in chosen_components:
                check.setChecked(True)
            check.stateChanged.connect(self.check_change)
            check.mouse_enter.connect(self.mouse_enter.emit)
            check.mouse_leave.connect(self.mouse_leave.emit)
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
        # TODO Check what situation create report of id ID: af9b57f074264169b4353aa1e61d8bc2
        if num >= len(self.check_box):
            return False
        return self.check_box[num].isChecked()

    def get_chosen(self):
        return [num for num, check in self.check_box.items() if check.isChecked()]

    def get_mask(self):
        res = [0]
        for _, check in sorted(self.check_box.items()):
            res.append(check.isChecked())
        return np.array(res, dtype=np.uint8)


class AlgorithmOptions(QWidget):
    def __init__(self, settings: StackSettings, image_view: StackImageView):
        control_view = image_view.get_control_view()
        super().__init__()
        self.settings = settings
        self.show_result = EnumComboBox(LabelEnum)  # QCheckBox("Show result")
        self.show_result.set_value(control_view.show_label)
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
        # noinspection PyUnresolvedReferences
        self.borders_thick.valueChanged.connect(self.border_value_check)
        self.execute_in_background_btn = QPushButton("Execute in background")
        self.execute_in_background_btn.setToolTip("Run calculation in background. Put result in multiple files panel")
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setStyleSheet("QPushButton{font-weight: bold;}")
        self.execute_all_btn = QPushButton("Execute all")
        self.execute_all_btn.setToolTip(
            "Execute in batch mode segmentation with current parameter. " "File list need to be specified in image tab."
        )
        self.execute_all_btn.setDisabled(True)
        self.save_parameters_btn = QPushButton("Save parameters")
        self.block_execute_all_btn = False
        self.algorithm_choose_widget = AlgorithmChoose(settings, mask_algorithm_dict)
        self.algorithm_choose_widget.result.connect(self.execution_result_set)
        self.algorithm_choose_widget.finished.connect(self.execution_finished)
        self.algorithm_choose_widget.progress_signal.connect(self.progress_info)

        # self.stack_layout = QStackedLayout()
        self.keep_chosen_components_chk = QCheckBox("Save selected components")
        self.keep_chosen_components_chk.setToolTip(
            "Save chosen components when loading segmentation form file\n" "or from multiple file widget."
        )
        self.keep_chosen_components_chk.stateChanged.connect(self.set_keep_chosen_components)
        self.keep_chosen_components_chk.setChecked(settings.keep_chosen_components)
        self.show_parameters = QPushButton("Show parameters")
        self.show_parameters.setToolTip("Show parameters of segmentation for each components")
        self.show_parameters_widget = SegmentationInfoDialog(
            self.settings, self.algorithm_choose_widget.change_algorithm
        )
        self.show_parameters.clicked.connect(self.show_parameters_widget.show)
        self.choose_components = ChosenComponents()
        self.choose_components.check_change_signal.connect(control_view.components_change)
        self.choose_components.mouse_leave.connect(image_view.component_unmark)
        self.choose_components.mouse_enter.connect(image_view.component_mark)
        # WARNING works only with one channels algorithms
        # SynchronizeValues.add_synchronization("channels_chose", widgets_list)
        self.chosen_list = []
        self.progress_bar2 = QProgressBar()
        self.progress_bar2.setHidden(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setHidden(True)
        self.progress_info_lab = QLabel()
        self.progress_info_lab.setHidden(True)
        self.file_list = []
        self.batch_process = BatchProceed()
        self.batch_process.progress_signal.connect(self.progress_info)
        self.batch_process.error_signal.connect(self.execution_all_error)
        self.batch_process.execution_done.connect(self.execution_all_done)
        self.batch_process.range_signal.connect(self.progress_bar.setRange)
        self.is_batch_process = False

        self.setContentsMargins(0, 0, 0, 0)
        main_layout = QVBoxLayout()
        # main_layout.setSpacing(0)
        opt_layout = QHBoxLayout()
        opt_layout.setContentsMargins(0, 0, 0, 0)
        opt_layout.addWidget(self.show_result)
        opt_layout.addWidget(right_label("Opacity:"))
        opt_layout.addWidget(self.opacity)
        main_layout.addLayout(opt_layout)
        opt_layout2 = QHBoxLayout()
        opt_layout2.setContentsMargins(0, 0, 0, 0)
        opt_layout2.addWidget(self.only_borders)
        opt_layout2.addWidget(right_label("Border thick:"))
        opt_layout2.addWidget(self.borders_thick)
        main_layout.addLayout(opt_layout2)
        btn_layout = QGridLayout()
        btn_layout.setContentsMargins(0, 0, 0, 0)
        btn_layout.addWidget(self.execute_btn, 0, 0)
        btn_layout.addWidget(self.execute_in_background_btn, 0, 1)
        btn_layout.addWidget(self.execute_all_btn, 1, 0)
        btn_layout.addWidget(self.save_parameters_btn, 1, 1)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.progress_bar2)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.progress_info_lab)
        main_layout.addWidget(self.algorithm_choose_widget, 1)
        # main_layout.addWidget(self.algorithm_choose)
        # main_layout.addLayout(self.stack_layout, 1)
        main_layout.addWidget(self.choose_components)
        down_layout = QHBoxLayout()
        down_layout.addWidget(self.keep_chosen_components_chk)
        down_layout.addWidget(self.show_parameters)
        main_layout.addLayout(down_layout)
        main_layout.addStretch()
        main_layout.setContentsMargins(0, 0, 0, 0)
        # main_layout.setSpacing(0)
        self.setLayout(main_layout)

        # noinspection PyUnresolvedReferences
        self.execute_in_background_btn.clicked.connect(self.execute_in_background)
        self.execute_btn.clicked.connect(self.execute_action)
        self.execute_all_btn.clicked.connect(self.execute_all_action)
        self.save_parameters_btn.clicked.connect(self.save_parameters)
        # noinspection PyUnresolvedReferences
        self.opacity.valueChanged.connect(control_view.set_opacity)
        # noinspection PyUnresolvedReferences
        self.show_result.current_choose.connect(control_view.set_show_label)
        self.only_borders.stateChanged.connect(control_view.set_borders)
        # noinspection PyUnresolvedReferences
        self.borders_thick.valueChanged.connect(control_view.set_borders_thick)
        image_view.component_clicked.connect(self.choose_components.other_component_choose)
        settings.chosen_components_widget = self.choose_components
        settings.components_change_list.connect(self.choose_components.new_choose)
        settings.image_changed.connect(self.choose_components.remove_components)

    @Slot(int)
    def set_keep_chosen_components(self, val):
        self.settings.set_keep_chosen_components(val)

    def save_parameters(self):
        dial = SaveDialog(io_functions.save_parameters_dict, False, history=self.settings.get_path_history())
        if not dial.exec_():
            return
        res = dial.get_result()
        self.settings.add_path_history(os.path.dirname(str(res.save_destination)))
        res.save_class.save(res.save_destination, self.algorithm_choose_widget.current_parameters())

    def border_value_check(self, value):
        if value % 2 == 0:
            self.borders_thick.setValue(value + 1)

    def file_list_change(self, val):
        self.file_list = val
        if len(self.file_list) > 0 and not self.block_execute_all_btn:
            self.execute_all_btn.setEnabled(True)
        else:
            self.execute_all_btn.setDisabled(True)

    def get_chosen_components(self):
        return sorted(self.choose_components.get_chosen())

    @property
    def segmentation(self):
        return self.settings.roi

    @segmentation.setter
    def segmentation(self, val):
        self.settings.roi = val

    def _image_changed(self):
        self.settings.roi = None
        self.choose_components.set_chose([], [])

    def _execute_in_background_init(self):
        if self.batch_process.isRunning():
            return
        self.progress_bar2.setVisible(True)
        self.progress_bar2.setRange(0, self.batch_process.queue.qsize())
        self.progress_bar2.setValue(self.batch_process.index)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.execute_btn.setDisabled(True)
        self.batch_process.start()

    def execute_in_background(self):
        # TODO check if components are properly passed
        widget = self.algorithm_choose_widget.current_widget()
        segmentation_profile = widget.get_segmentation_profile()
        task = BatchTask(self.settings.get_project_info(), segmentation_profile, None)
        self.batch_process.add_task(task)
        self.progress_bar2.setRange(0, self.progress_bar2.maximum() + 1)
        self._execute_in_background_init()

    def execute_all_action(self):
        dial = SaveDialog(
            {SaveROI.get_name(): SaveROI},
            history=self.settings.get_path_history(),
            system_widget=False,
        )
        dial.setFileMode(QFileDialog.Directory)
        dial.setDirectory(self.settings.get("io.save_batch", self.settings.get("io.save_segmentation_directory", "")))
        if not dial.exec_():
            return
        folder_path = str(dial.selectedFiles()[0])
        self.settings.set("io.save_batch", folder_path)

        widget = self.algorithm_choose_widget.current_widget()

        save_parameters = dial.values
        segmentation_profile = widget.get_segmentation_profile()
        for file_path in self.file_list:
            task = BatchTask(file_path, segmentation_profile, (folder_path, save_parameters))
            self.batch_process.add_task(task)
        self.progress_bar2.setRange(0, self.progress_bar2.maximum() + len(self.file_list))
        self._execute_in_background_init()

    def execution_all_error(self, text):
        QMessageBox.warning(self, "Proceed error", text)

    def execution_all_done(self):
        if not self.batch_process.queue.empty():
            self._execute_in_background_init()
            return
        self.execute_btn.setEnabled(True)
        self.block_execute_all_btn = False
        if len(self.file_list) > 0:
            self.execute_all_btn.setEnabled(True)
        self.progress_bar.setHidden(True)
        self.progress_bar2.setHidden(True)
        self.progress_info_lab.setHidden(True)

    def execute_action(self):
        self.execute_btn.setDisabled(True)
        self.execute_all_btn.setDisabled(True)
        self.block_execute_all_btn = True
        self.is_batch_process = False
        self.progress_bar.setRange(0, 0)
        self.choose_components.setDisabled(True)
        chosen = sorted(self.choose_components.get_chosen())
        blank = get_mask(self.settings.roi, self.settings.mask, chosen)
        if blank is not None:
            # Problem with handling time data in algorithms
            # TODO Fix This
            blank = blank[0]
        self.progress_bar.setHidden(False)
        widget: AlgorithmSettingsWidget = self.algorithm_choose_widget.current_widget()
        widget.set_mask(blank)
        self.progress_bar.setRange(0, widget.algorithm.get_steps_num())
        widget.execute()
        self.chosen_list = chosen

    def progress_info(self, text, num, file_name="", file_num=0):
        self.progress_info_lab.setVisible(True)
        if file_name != "":
            self.progress_info_lab.setText(file_name + "\n" + text)
        else:
            self.progress_info_lab.setText(text)
        self.progress_bar.setValue(num)
        self.progress_bar2.setValue(file_num)

    def execution_finished(self):
        self.execute_btn.setEnabled(True)
        self.block_execute_all_btn = False
        if len(self.file_list) > 0:
            self.execute_all_btn.setEnabled(True)
        self.progress_bar.setHidden(True)
        self.progress_info_lab.setHidden(True)
        self.choose_components.setDisabled(False)

    def execution_result_set(self, result):
        self.settings.set_segmentation_result(result)

    def showEvent(self, _):
        widget = self.algorithm_choose_widget.current_widget()
        widget.image_changed(self.settings.image)


class ImageInformation(QWidget):
    def __init__(self, settings: StackSettings, parent=None):
        """:type settings: ImageSettings"""
        super().__init__(parent)
        self._settings = settings
        self.path = QTextEdit("<b>Path:</b> example image")
        self.path.setWordWrapMode(QTextOption.WrapAnywhere)
        self.path.setReadOnly(True)
        self.setMinimumHeight(20)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        self.multiple_files = QCheckBox("Show multiple files panel")
        self.multiple_files.setChecked(settings.get("multiple_files_widget", True))
        self.multiple_files.stateChanged.connect(self.set_multiple_files)
        units_value = self._settings.get("units_value", Units.nm)
        for el in self.spacing:
            el.setAlignment(Qt.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.NoButtons)
            el.setRange(0, 100000)
            # noinspection PyUnresolvedReferences
            el.valueChanged.connect(self.image_spacing_change)
        self.units = EnumComboBox(Units)
        self.units.set_value(units_value)
        # noinspection PyUnresolvedReferences
        self.units.currentIndexChanged.connect(self.update_spacing)

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
        layout.addStretch(1)
        layout.addWidget(self.multiple_files)
        self.setLayout(layout)
        self._settings.image_changed[str].connect(self.set_image_path)

    @Slot(int)
    def set_multiple_files(self, val):
        self._settings.set("multiple_files_widget", val)

    def update_spacing(self, index=None):
        units_value = self.units.get_value()
        if index is not None:
            self._settings.set("units_value", units_value)
        for el, val in zip(self.spacing, self._settings.image_spacing[::-1]):
            el.blockSignals(True)
            el.setValue(val * UNIT_SCALE[units_value.value])
            el.blockSignals(False)
        if self._settings.is_image_2d():
            # self.spacing[2].setValue(0)
            self.spacing[2].setDisabled(True)
        else:
            self.spacing[2].setDisabled(False)

    def set_image_path(self, value):
        self.path.setText(f"<b>Path:</b> {value}")
        self.update_spacing()

    def image_spacing_change(self):
        self._settings.image_spacing = [
            el.value() / UNIT_SCALE[self.units.currentIndex()] for i, el in enumerate(self.spacing[::-1])
        ]

    def showEvent(self, _a0):
        units_value = self._settings.get("units_value", Units.nm)
        for el, val in zip(self.spacing, self._settings.image_spacing[::-1]):
            el.setValue(val * UNIT_SCALE[units_value.value])
        if self._settings.is_image_2d():
            self.spacing[2].setValue(0)
            self.spacing[2].setDisabled(True)
        else:
            self.spacing[2].setDisabled(False)


class Options(QTabWidget):
    def __init__(self, settings, image_view, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.algorithm_options = AlgorithmOptions(settings, image_view)
        self.image_properties = ImageInformation(settings, parent)
        self.image_properties.add_files.file_list_changed.connect(self.algorithm_options.file_list_change)
        self.algorithm_options.batch_process.multiple_result.connect(
            partial(self.image_properties.multiple_files.setChecked, True)
        )
        self.addTab(self.image_properties, "Image")
        self.addTab(self.algorithm_options, "Segmentation")
        self.setMinimumWidth(370)
        self.setCurrentIndex(1)

    def get_chosen_components(self):
        return self.algorithm_options.get_chosen_components()


class MainWindow(BaseMainWindow):
    @classmethod
    def get_setting_class(cls) -> Type[StackSettings]:
        return StackSettings

    initial_image_path = PartSegData.segmentation_mask_default_image

    def __init__(
        self, config_folder=CONFIG_FOLDER, title="PartSeg", settings=None, signal_fun=None, initial_image=None
    ):
        super().__init__(config_folder, title, settings, io_functions.load_dict, signal_fun)
        self.channel_info = "channelcontrol"
        self.channel_control = ChannelProperty(self.settings, start_name="channelcontrol")
        self.image_view = StackImageView(self.settings, self.channel_control, name="channelcontrol")
        self.image_view.setMinimumWidth(450)
        self.info_text = QLabel()
        self.info_text.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.image_view.text_info_change.connect(self.info_text.setText)
        self.options_panel = Options(self.settings, self.image_view)
        self.main_menu = MainMenu(self.settings, self)
        self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)
        self.color_bar = ColorBar(self.settings, self.image_view)
        self.multiple_files = MultipleFileWidget(self.settings, io_functions.load_dict)
        self.multiple_files.setVisible(self.options_panel.image_properties.multiple_files.isChecked())
        self.options_panel.algorithm_options.batch_process.multiple_result.connect(
            partial(self.multiple_files.save_state_action, custom_name=False)
        )

        icon = QIcon(os.path.join(PartSegData.icons_dir, "icon_stack.png"))
        self.setWindowIcon(icon)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("&Open").triggered.connect(self.main_menu.load_image)
        file_menu.addMenu(self.recent_file_menu)
        file_menu.addAction("&Save segmentation").triggered.connect(self.main_menu.save_segmentation)
        file_menu.addAction("&Save components").triggered.connect(self.main_menu.save_result)
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Settings and Measurement").triggered.connect(self.main_menu.show_advanced_window)
        view_menu.addAction("Additional output").triggered.connect(self.additional_layers_show)
        view_menu.addAction("Additional output with data").triggered.connect(lambda: self.additional_layers_show(True))
        view_menu.addAction("Napari viewer").triggered.connect(self.napari_viewer_show)
        view_menu.addAction("Toggle Multiple Files").triggered.connect(self.toggle_multiple_files)
        action = view_menu.addAction("Screenshot")
        action.triggered.connect(self.screenshot(self.image_view))
        action.setShortcut(QKeySequence.Print)
        image_menu = menu_bar.addMenu("Image operations")
        image_menu.addAction("Image adjustment").triggered.connect(self.image_adjust_exec)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("State directory").triggered.connect(self.show_settings_directory)
        help_menu.addAction("About").triggered.connect(self.show_about_dialog)

        layout = QVBoxLayout()
        layout.addWidget(self.main_menu)
        sub_layout = QHBoxLayout()
        sub2_layout = QVBoxLayout()
        sub3_layout = QVBoxLayout()
        sub_layout.addWidget(self.multiple_files)
        sub_layout.addWidget(self.color_bar, 0)
        sub3_layout.addWidget(self.image_view, 1)
        sub3_layout.addWidget(self.info_text, 0)
        sub2_layout.addWidget(self.options_panel, 1)
        sub2_layout.addWidget(self.channel_control, 0)

        sub_layout.addLayout(sub3_layout, 1)
        sub_layout.addLayout(sub2_layout, 0)
        layout.addLayout(sub_layout)
        self.widget = QWidget()
        self.widget.setLayout(layout)
        self.setCentralWidget(self.widget)
        if initial_image is None:
            reader = TiffImageReader()
            im = reader.read(self.initial_image_path)
            im.file_path = ""
            self.settings.image = im
        elif initial_image is False:
            # FIXME This is for test opening
            pass
        else:
            self.settings.image = initial_image
        try:
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))
        except KeyError:
            pass

    def closeEvent(self, event: QCloseEvent):
        self.settings.set_in_profile("main_window_geometry", self.saveGeometry().toHex().data().decode("ascii"))
        self.options_panel.algorithm_options.algorithm_choose_widget.recursive_get_values()
        self.main_menu.segmentation_dialog.close()
        self.options_panel.algorithm_options.show_parameters_widget.close()
        if self.main_menu.advanced_window is not None:
            self.main_menu.advanced_window.close()
            del self.main_menu.advanced_window
        if self.main_menu.measurements_window is not None:
            self.main_menu.measurements_window.close()
            del self.main_menu.measurements_window
        del self.main_menu.segmentation_dialog
        del self.options_panel.algorithm_options.show_parameters_widget
        self.settings.dump()
        super().closeEvent(event)

    @staticmethod
    def get_project_info(file_path, image):
        return MaskProjectTuple(file_path=file_path, image=image)

    def set_data(self, data):
        self.main_menu.set_data(data)

    def change_theme(self):
        self.image_view.set_theme(self.settings.theme_name)
        super().change_theme()
