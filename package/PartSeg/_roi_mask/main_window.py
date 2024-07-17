import os
from contextlib import suppress
from functools import partial
from typing import Type

import numpy as np
from qtpy.QtCore import QByteArray, Qt, Signal, Slot
from qtpy.QtGui import QCloseEvent, QGuiApplication, QIcon, QKeySequence, QTextOption
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDoubleSpinBox,
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
from superqt import QEnumComboBox

import PartSegData
from PartSeg import state_store
from PartSeg._roi_mask.batch_proceed import BatchProceed, BatchTask
from PartSeg._roi_mask.image_view import StackImageView
from PartSeg._roi_mask.segmentation_info_dialog import SegmentationInfoDialog
from PartSeg._roi_mask.simple_measurements import SimpleMeasurements
from PartSeg._roi_mask.stack_settings import StackSettings, get_mask
from PartSeg.common_backend.base_settings import IO_SAVE_DIRECTORY, ROI_NOT_FIT
from PartSeg.common_gui.advanced_tabs import AdvancedWindow, ImageMetadata
from PartSeg.common_gui.algorithms_description import AlgorithmChoose, InteractiveAlgorithmSettingsWidget
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.custom_load_dialog import PLoadDialog
from PartSeg.common_gui.custom_save_dialog import PSaveDialog
from PartSeg.common_gui.exception_hooks import load_data_exception_hook
from PartSeg.common_gui.flow_layout import FlowLayout
from PartSeg.common_gui.main_window import OPEN_DIRECTORY, OPEN_FILE, OPEN_FILE_FILTER, BaseMainMenu, BaseMainWindow
from PartSeg.common_gui.mask_widget import MaskDialogBase
from PartSeg.common_gui.multiple_file_widget import MultipleFileWidget
from PartSeg.common_gui.napari_image_view import LabelEnum
from PartSeg.common_gui.select_multiple_files import AddFiles
from PartSeg.common_gui.stack_image_view import ColorBar
from PartSeg.common_gui.universal_gui_part import right_label
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore import UNIT_SCALE, Units
from PartSegCore.io_utils import WrongFileTypeException
from PartSegCore.mask import io_functions
from PartSegCore.mask.algorithm_description import MaskAlgorithmSelection
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import LoadROI, LoadROIFromTIFF, LoadROIParameters, MaskProjectTuple, SaveROI
from PartSegCore.project_info import HistoryElement, HistoryProblem, calculate_mask_from_project
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image, TiffImageReader

CONFIG_FOLDER = os.path.join(state_store.save_folder, "mask")


class MaskDialog(MaskDialogBase):
    # FIXME add tests
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
        self.settings._set_roi_info(  # pylint: disable=protected-access
            ROIInfo(seg["segmentation"]),
            False,
            history.roi_extraction_parameters["selected"],
            history.roi_extraction_parameters["parameters"],
        )
        self.settings.mask = seg.get("mask")
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
            self.advanced_window = AdvancedWindow(self.settings, ["channelcontrol"], reload_list=[self.reload])
        self.advanced_window.show()

    def reload(self):
        self.parent().parent().options_panel.algorithm_options.algorithm_choose_widget.reload(MaskAlgorithmSelection)

    def load_image(self):
        # TODO move segmentation with image load to load_segmentaion
        dial = PLoadDialog(
            io_functions.load_dict,
            settings=self.settings,
            path=OPEN_DIRECTORY,
            filter_path=OPEN_FILE_FILTER,
        )
        default_file_path = self.settings.get(OPEN_FILE, "")
        if os.path.isfile(default_file_path):
            dial.selectFile(default_file_path)
        if not dial.exec_():
            return
        load_property = dial.get_result()
        self.settings.set(OPEN_FILE, load_property.load_location[0])
        self.settings.add_last_files(load_property.load_location, load_property.load_class.get_name())

        execute_dialog = ExecuteFunctionDialog(
            load_property.load_class.load,
            [load_property.load_location],
            {"metadata": {"default_spacing": self.settings.image.spacing}},
            text="Load data",
            exception_hook=load_data_exception_hook,
        )
        if execute_dialog.exec_():
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
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if res == QMessageBox.StandardButton.Yes:
                image = image.swap_time_and_stack()
            else:
                return False
        self.settings.image = image
        return True

    def load_segmentation(self):
        settings_path = OPEN_DIRECTORY if self.settings.get("sync_dirs", False) else "io.open_segmentation_directory"
        dial = PLoadDialog(
            {
                LoadROI.get_name(): LoadROI,
                LoadROIParameters.get_name(): LoadROIParameters,
                LoadROIFromTIFF.get_name(): LoadROIFromTIFF,
            },
            settings=self.settings,
            path=settings_path,
        )
        if not dial.exec_():
            return
        load_property = dial.get_result()

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
        if not dial.exec_():
            return
        result: MaskProjectTuple = dial.get_result()
        if result is None:
            QMessageBox.critical(self, "Data Load fail", "Fail of loading data")
            return
        if result.roi_info.roi is not None:
            try:
                self.settings.set_project_info(dial.get_result())
                return
            except ValueError as e:
                if e.args != (ROI_NOT_FIT,):
                    raise
                text = (
                    f"Segmentation of shape {result.roi_info.roi.shape}\n({result.file_path})\n"
                    f"do not fit to image of shape {self.settings.image.shape}\n({self.settings.image.file_path})"
                )
                if all(x is None for x in result.roi_extraction_parameters.values()):
                    QMessageBox.warning(self, "Segmentation do not fit", f"{text} and no parameters for extraction")
                    return

                self.segmentation_dialog.set_additional_text(text + "\nmaybe you would like to load parameters only.")
            except HistoryProblem:
                QMessageBox().warning(
                    self,
                    "Load Problem",
                    "You set to save selected components when loading "
                    "another segmentation but history is incompatibility",
                )

        else:
            self.segmentation_dialog.set_additional_text("")
        self.segmentation_dialog.set_parameters_dict(result.roi_extraction_parameters)
        self.segmentation_dialog.show()

    def save_segmentation(self):
        if self.settings.roi is None:
            QMessageBox.warning(self, "No segmentation", "No segmentation to save")
            return
        settings_path = OPEN_DIRECTORY if self.settings.get("sync_dirs", False) else "io.save_segmentation_directory"
        dial = PSaveDialog(
            io_functions.save_segmentation_dict,
            system_widget=False,
            settings=self.settings,
            path=settings_path,
        )

        dial.selectFile(f"{os.path.splitext(os.path.basename(self.settings.image_path))[0]}.seg")

        if not dial.exec_():
            return
        save_location, _selected_filter, save_class, values = dial.get_result()

        def exception_hook(exception):
            QMessageBox.critical(
                self, "Save error", f"Error on disc operation. Text: {exception}", QMessageBox.StandardButton.Ok
            )
            raise exception

        dial = ExecuteFunctionDialog(
            save_class.save,
            [save_location, self.settings.get_project_info(), values],
            text="Save segmentation",
            exception_hook=exception_hook,
        )
        dial.exec_()

    def save_result(self):
        if self.settings.image_path is not None and QMessageBox.StandardButton.Yes == QMessageBox.question(
            self,
            "Copy",
            "Copy name to clipboard?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        ):
            clipboard = QGuiApplication.clipboard()
            clipboard.setText(os.path.splitext(os.path.basename(self.settings.image_path))[0])

        if self.settings.roi is None or len(self.settings.sizes) == 1:
            QMessageBox.warning(self, "No components", "No components to save")
            return
        dial = PSaveDialog(
            io_functions.save_components_dict,
            system_widget=False,
            settings=self.settings,
            file_mode=PSaveDialog.Directory,
            path="io.save_components_directory",
        )
        dial.selectFile(os.path.splitext(os.path.basename(self.settings.image_path))[0])
        if not dial.exec_():
            return
        res = dial.get_result()
        potential_names = self.settings.get_file_names_for_save_result(res.save_destination)
        conflict = [el for el in potential_names if os.path.exists(el)]

        if conflict:
            # TODO modify because of long lists
            conflict_str = "\n".join(conflict)
            if QMessageBox.StandardButton.No == QMessageBox.warning(
                self,
                "Overwrite",
                f"Overwrite files:\n {conflict_str}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ):
                self.save_result()

        def exception_hook(exception):
            QMessageBox.critical(
                self, "Save error", f"Error on disc operation. Text: {exception}", QMessageBox.StandardButton.Ok
            )

        dial = ExecuteFunctionDialog(
            res.save_class.save,
            [res.save_destination, self.settings.get_project_info(), res.parameters],
            text="Save components",
            exception_hook=exception_hook,
        )
        dial.exec_()


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
    :type check_box: dict[int, ComponentCheckBox]
    """

    check_change_signal = Signal()
    mouse_enter = Signal(int)
    mouse_leave = Signal(int)

    def __init__(self):
        super().__init__()
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
        return False if num >= len(self.check_box) else self.check_box[num].isChecked()

    def get_chosen(self):
        return [num for num, check in self.check_box.items() if check.isChecked()]

    def get_mask(self):
        res = [0]
        res.extend(check.isChecked() for _, check in sorted(self.check_box.items()))
        return np.array(res, dtype=np.uint8)


class AlgorithmOptions(QWidget):
    def __init__(self, settings: StackSettings, image_view: StackImageView):  # noqa: PLR0915
        super().__init__()
        self.settings = settings
        self.view_name = image_view.name
        self.show_result = QEnumComboBox(enum_class=LabelEnum)  # QCheckBox("Show result")
        self._set_show_label_from_settings()
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setSingleStep(0.1)
        self._set_opacity_from_settings()
        self.only_borders = QCheckBox("Only borders")
        self._set_border_mode_from_settings()
        self.borders_thick = QSpinBox()
        self.borders_thick.setRange(1, 25)
        self.borders_thick.setSingleStep(1)
        self._set_border_thick_from_settings()
        self.execute_in_background_btn = QPushButton("Execute in background")
        self.execute_in_background_btn.setToolTip("Run calculation in background. Put result in multiple files panel")
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.setStyleSheet("QPushButton{font-weight: bold;}")
        self.execute_all_btn = QPushButton("Execute all")
        self.execute_all_btn.setToolTip(
            "Execute in batch mode segmentation with current parameter. File list need to be specified in image tab."
        )
        self.execute_all_btn.setDisabled(True)
        self.save_parameters_btn = QPushButton("Save parameters")
        self.block_execute_all_btn = False
        self.algorithm_choose_widget = AlgorithmChoose(settings, MaskAlgorithmSelection)
        self.algorithm_choose_widget.result.connect(self.execution_result_set)
        self.algorithm_choose_widget.finished.connect(self.execution_finished)
        self.algorithm_choose_widget.progress_signal.connect(self.progress_info)

        self.keep_chosen_components_chk = QCheckBox("Save selected components")
        self.keep_chosen_components_chk.setToolTip(
            "Save chosen components when loading segmentation form file\n or from multiple file widget."
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
        self.choose_components.check_change_signal.connect(image_view.refresh_selected)
        self.choose_components.mouse_leave.connect(image_view.component_unmark)
        self.choose_components.mouse_enter.connect(image_view.component_mark)
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
        main_layout.addWidget(self.choose_components)
        down_layout = QHBoxLayout()
        down_layout.addWidget(self.keep_chosen_components_chk)
        down_layout.addWidget(self.show_parameters)
        main_layout.addLayout(down_layout)
        main_layout.addStretch()
        main_layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(main_layout)

        # noinspection PyUnresolvedReferences
        self.execute_in_background_btn.clicked.connect(self.execute_in_background)
        self.execute_btn.clicked.connect(self.execute_action)
        self.execute_all_btn.clicked.connect(self.execute_all_action)
        self.save_parameters_btn.clicked.connect(self.save_parameters)
        # noinspection PyUnresolvedReferences
        self.opacity.valueChanged.connect(self._set_opacity)
        # noinspection PyUnresolvedReferences
        self.show_result.currentEnumChanged.connect(self._set_show_label)
        self.only_borders.stateChanged.connect(self._set_border_mode)
        # noinspection PyUnresolvedReferences
        self.borders_thick.valueChanged.connect(self._set_border_thick)
        image_view.component_clicked.connect(self.choose_components.other_component_choose)
        settings.chosen_components_widget = self.choose_components
        settings.components_change_list.connect(self.choose_components.new_choose)
        settings.image_changed.connect(self.choose_components.remove_components)
        settings.connect_to_profile(f"{self.view_name}.image_state.only_border", self._set_border_mode_from_settings)
        settings.connect_to_profile(f"{self.view_name}.image_state.border_thick", self._set_border_thick_from_settings)
        settings.connect_to_profile(f"{self.view_name}.image_state.opacity", self._set_opacity_from_settings)
        settings.connect_to_profile(f"{self.view_name}.image_state.show_label", self._set_show_label_from_settings)

    def _set_border_mode(self, value: bool):
        self.settings.set_in_profile(f"{self.view_name}.image_state.only_border", value)

    def _set_border_thick(self, value: int):
        self.settings.set_in_profile(f"{self.view_name}.image_state.border_thick", value)

    def _set_opacity(self, value: float):
        self.settings.set_in_profile(f"{self.view_name}.image_state.opacity", value)

    def _set_show_label(self, value: LabelEnum):
        self.settings.set_in_profile(f"{self.view_name}.image_state.show_label", value)

    def _set_border_mode_from_settings(self):
        self.only_borders.setChecked(self.settings.get_from_profile(f"{self.view_name}.image_state.only_border", True))

    def _set_border_thick_from_settings(self):
        self.borders_thick.setValue(self.settings.get_from_profile(f"{self.view_name}.image_state.border_thick", 1))

    def _set_opacity_from_settings(self):
        self.opacity.setValue(self.settings.get_from_profile(f"{self.view_name}.image_state.opacity", 1.0))

    def _set_show_label_from_settings(self):
        self.show_result.setCurrentEnum(
            self.settings.get_from_profile(f"{self.view_name}.image_state.show_label", LabelEnum.Show_results)
        )

    @Slot(int)
    def set_keep_chosen_components(self, val):
        self.settings.set_keep_chosen_components(val)

    def save_parameters(self):
        dial = PSaveDialog(
            io_functions.save_parameters_dict, system_widget=False, settings=self.settings, path=IO_SAVE_DIRECTORY
        )
        if not dial.exec_():
            return
        res = dial.get_result()
        res.save_class.save(
            save_location=res.save_destination,
            project_info=self.algorithm_choose_widget.current_parameters(),
        )

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
        dial = PSaveDialog(
            SaveROI,
            settings=self.settings,
            system_widget=False,
            path="io.save_batch",
            file_mode=PSaveDialog.Directory,
        )
        if not dial.exec_():
            return
        folder_path = str(dial.selectedFiles()[0])

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
        widget: InteractiveAlgorithmSettingsWidget = self.algorithm_choose_widget.current_widget()
        widget.set_mask(blank)
        self.progress_bar.setRange(0, widget.algorithm.get_steps_num())
        widget.execute()
        self.chosen_list = chosen

    def progress_info(self, text, num, file_name="", file_num=0):
        self.progress_info_lab.setVisible(True)
        if file_name:
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
        self.path.setWordWrapMode(QTextOption.WrapMode.WrapAnywhere)
        self.path.setReadOnly(True)
        self.setMinimumHeight(20)
        self.spacing = [QDoubleSpinBox() for _ in range(3)]
        self.multiple_files = QCheckBox("Show multiple files panel")
        self.multiple_files.setChecked(settings.get("multiple_files_widget", True))
        self.multiple_files.stateChanged.connect(self.set_multiple_files)
        self.sync_dirs = QCheckBox("Sync directories in file dialog")
        self.sync_dirs.setToolTip(
            "If checked then 'Load Image', 'Load segmentation' and 'Save segmentation' "
            "will open file dialog in the same directory"
        )
        self.sync_dirs.setChecked(settings.get("sync_dirs", False))
        self.sync_dirs.stateChanged.connect(self.set_sync_dirs)
        units_value = self._settings.get("units_value", Units.nm)
        for el in self.spacing:
            el.setAlignment(Qt.AlignmentFlag.AlignRight)
            el.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
            el.setRange(0, 100000)
            # noinspection PyUnresolvedReferences
            el.valueChanged.connect(self.image_spacing_change)
        self.units = QEnumComboBox(enum_class=Units)
        self.units.setCurrentEnum(units_value)
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
        layout.addWidget(self.sync_dirs)
        self.setLayout(layout)
        self._settings.image_path_changed.connect(self.set_image_path)
        self._settings.connect_("multiple_files_widget", self._set_multiple_files)
        self._settings.connect_("sync_dirs", self._set_sync_dirs)

    @Slot(int)
    def set_multiple_files(self, val):
        self._settings.set("multiple_files_widget", val)

    @Slot(int)
    def set_sync_dirs(self, val):
        self._settings.set("sync_dirs", val)

    def _set_multiple_files(self):
        self.multiple_files.setChecked(self._settings.get("multiple_files_widget", True))

    def _set_sync_dirs(self):
        self.sync_dirs.setChecked(self._settings.get("sync_dirs", False))

    def update_spacing(self, index=None):
        units_value = self.units.currentEnum()
        if index is not None:
            self._settings.set("units_value", units_value)
        for el, val in zip(self.spacing, self._settings.image_spacing[::-1]):
            el.blockSignals(True)
            el.setValue(val * UNIT_SCALE[units_value.value])
            el.blockSignals(False)
        if self._settings.is_image_2d():
            self.spacing[2].setDisabled(True)
        else:
            self.spacing[2].setDisabled(False)

    def set_image_path(self, value):
        self.path.setText(f"<b>Path:</b> {value}")
        self.update_spacing()

    def image_spacing_change(self):
        self._settings.image_spacing = [el.value() / UNIT_SCALE[self.units.currentIndex()] for el in self.spacing[::-1]]

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
        self.image_properties = ImageInformation(settings)
        self.image_metadata = ImageMetadata(settings)
        self.image_properties.add_files.file_list_changed.connect(self.algorithm_options.file_list_change)
        self.algorithm_options.batch_process.multiple_result.connect(
            partial(self.image_properties.multiple_files.setChecked, True)
        )
        self.addTab(self.image_properties, "Image")
        self.addTab(self.image_metadata, "Image metadata")
        self.addTab(self.algorithm_options, "Segmentation")
        self.setMinimumWidth(370)
        self.setCurrentIndex(2)

    def get_chosen_components(self):
        return self.algorithm_options.get_chosen_components()


class MainWindow(BaseMainWindow):
    settings: StackSettings

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
        self.info_text.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
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

        self._setup_menu_bar()

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
        elif initial_image is not False:
            self.settings.image = initial_image
        with suppress(KeyError):
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))

    def _setup_menu_bar(self):
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
        view_menu.addAction("Toggle console").triggered.connect(self._toggle_console)
        view_menu.addAction("Toggle scale bar").triggered.connect(self._toggle_scale_bar)
        action = view_menu.addAction("Screenshot")
        action.triggered.connect(self.screenshot(self.image_view))
        action.setShortcut(QKeySequence.StandardKey.Print)
        image_menu = menu_bar.addMenu("Image operations")
        image_menu.addAction("Image adjustment").triggered.connect(self.image_adjust_exec)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("State directory").triggered.connect(self.show_settings_directory)
        help_menu.addAction("About").triggered.connect(self.show_about_dialog)

    def _toggle_scale_bar(self):
        self.image_view.toggle_scale_bar()
        super()._toggle_scale_bar()

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
        super().closeEvent(event)

    @staticmethod
    def get_project_info(file_path, image, roi_info=None):
        if roi_info is None:
            roi_info = ROIInfo(None)
        return MaskProjectTuple(
            file_path=file_path,
            image=image,
            roi_info=roi_info,
            roi_extraction_parameters={i: None for i in roi_info.bound_info},
        )

    def set_data(self, data):
        self.main_menu.set_data(data)

    # @ensure_main_thread
    def change_theme(self, event):
        self.image_view.set_theme(self.settings.theme_name)
        super().change_theme(event)
