import os
import sys
from pathlib import Path
from typing import Type

import PartSegData
import numpy as np
from qtpy.QtCore import Qt, QByteArray, QEvent
from qtpy.QtGui import QIcon, QKeyEvent, QKeySequence, QResizeEvent
from qtpy.QtWidgets import (
    QLabel,
    QWidget,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QMessageBox,
    QCheckBox,
    QComboBox,
    QInputDialog,
)

from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.stacked_widget_with_selector import StackedWidgetWithSelector
from PartSeg.segmentation_analysis.measurement_widget import MeasurementWidget
from PartSegCore.analysis import ProjectTuple
from PartSegCore.analysis import algorithm_description, load_functions
from PartSegCore.analysis.io_utils import create_history_element_from_project
from PartSegCore.io_utils import WrongFileTypeException, HistoryElement
from PartSegCore.mask_create import calculate_mask
from ..common_gui.algorithms_description import InteractiveAlgorithmSettingsWidget, AlgorithmChoose
from ..common_gui.channel_control import ChannelProperty
from ..common_gui.equal_column_layout import EqualColumnLayout
from ..common_gui.mask_widget import MaskDialogBase
from ..common_gui.stack_image_view import ColorBar
from ..common_gui.universal_gui_part import TextShow
from ..common_gui.waiting_dialog import WaitingDialog, ExecuteFunctionDialog
from ..common_gui.multiple_file_widget import MultipleFileWidget
from PartSegCore.segmentation.algorithm_base import SegmentationResult
from PartSeg.common_gui.main_window import BaseMainWindow, BaseMainMenu
from .advanced_window import SegAdvancedWindow
from .batch_window import BatchWindow
from .calculation_pipeline_thread import CalculatePipelineThread
from PartSegImage import TiffImageReader
from PartSegCore.algorithm_describe_base import SegmentationProfile
from PartSegCore.analysis.analysis_utils import SegmentationPipelineElement, SegmentationPipeline
from .image_view import SynchronizeView, ResultImageView, CompareImageView
from .partseg_settings import PartSettings
from ..common_gui.custom_save_dialog import SaveDialog
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore import state_store

CONFIG_FOLDER = os.path.join(state_store.save_folder, "analysis")


class Options(QWidget):
    def __init__(
        self,
        settings: PartSettings,
        channel_control2: ChannelProperty,
        left_image: ResultImageView,
        main_image: ResultImageView,
        synchronize: SynchronizeView,
    ):
        super().__init__()
        self._settings = settings
        self.left_panel = left_image
        self._ch_control2 = channel_control2
        self.synchronize_val = False
        self.hide_left_panel_chk = QCheckBox("Hide left panel")
        self.hide_left_panel_chk.stateChanged.connect(self.hide_left_panel)
        self.synchronize_checkbox = QCheckBox("Synchronize view")
        self.synchronize_checkbox.stateChanged.connect(synchronize.set_synchronize)
        self.interactive_use = QCheckBox("Interactive use")
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self.execute_algorithm)
        self.execute_btn.setStyleSheet("QPushButton{font-weight: bold;}")
        self.save_pipe_btn = QPushButton("Save pipeline")
        self.save_pipe_btn.clicked.connect(self.save_pipeline)
        self.save_pipe_btn.setToolTip("Save current pipeline. Last element is last executed algorithm")
        self.choose_pipe = QComboBox()
        self.choose_pipe.addItem("<none>")
        self.choose_pipe.addItems(list(self._settings.segmentation_pipelines.keys()))
        self.choose_pipe.currentTextChanged.connect(self.choose_pipeline)
        self.choose_pipe.setToolTip("Execute chosen pipeline")
        self.save_profile_btn = QPushButton("Save profile")
        self.save_profile_btn.setToolTip("Save values from current view")
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.choose_profile = QComboBox()
        self.choose_profile.addItem("<none>")
        self.choose_profile.addItems(list(self._settings.segmentation_profiles.keys()))
        self.choose_profile.setToolTip("Select profile to restore its settings. Execute if interactive is checked")
        # image state
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setDisabled(True)
        self.compare_btn.clicked.connect(self.compare_action)
        left_image.hide_signal.connect(self.compare_btn.setHidden)

        self.update_tooltips()
        self.choose_profile.currentTextChanged.connect(self.change_profile)
        self.interactive_use.stateChanged.connect(self.execute_btn.setDisabled)
        self.interactive_use.stateChanged.connect(self.interactive_change)
        self.algorithm_choose_widget = AlgorithmChoose(settings, algorithm_description.analysis_algorithm_dict)
        self.algorithm_choose_widget.result.connect(self.execution_done)
        self.algorithm_choose_widget.finished.connect(self.calculation_finished)
        self.algorithm_choose_widget.value_changed.connect(self.interactive_algorithm_execute)
        self.algorithm_choose_widget.algorithm_changed.connect(self.interactive_algorithm_execute)

        self.label = TextShow()

        # self.label.setWordWrap(True)
        # self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout = QVBoxLayout()
        layout2 = QHBoxLayout()
        layout2.setSpacing(1)
        layout2.setContentsMargins(0, 0, 0, 0)
        layout3 = QHBoxLayout()
        layout3.setContentsMargins(0, 0, 0, 0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout5 = QHBoxLayout()
        layout5.setContentsMargins(0, 0, 0, 0)
        layout5.addWidget(self.save_pipe_btn)
        layout5.addWidget(self.choose_pipe)
        layout4 = QHBoxLayout()
        layout4.setContentsMargins(0, 0, 0, 0)
        layout4.addWidget(self.save_profile_btn)
        layout4.addWidget(self.choose_profile)
        layout3.addWidget(self.interactive_use)
        layout3.addWidget(self.execute_btn)
        layout.addLayout(layout5)
        layout.addLayout(layout4)
        layout.addLayout(layout3)
        layout.addWidget(self.algorithm_choose_widget, 1)
        # layout.addLayout(self.stack_layout)
        layout.addWidget(self.label)
        # layout.addStretch(1)
        layout2.addWidget(self.hide_left_panel_chk)
        layout2.addWidget(self.synchronize_checkbox)
        layout.addLayout(layout2)
        layout.addWidget(self._ch_control2)
        layout.setSpacing(0)
        self.setLayout(layout)

    def compare_action(self):
        if self.compare_btn.text() == "Compare":
            self._settings.set_segmentation_to_compare(self._settings.segmentation)
            self.compare_btn.setText("Remove")
        else:
            self._settings.set_segmentation_to_compare(None)
            self.compare_btn.setText("Compare")

    def calculation_finished(self):
        self.execute_btn.setDisabled(self.interactive_use.isChecked())
        self.interactive_use.setEnabled(True)

    def save_pipeline(self):
        history = self._settings.get_history()
        if not history:
            QMessageBox.information(self, "No mask created", "There is no new mask created", QMessageBox.Ok)
            return
        mask_history = []
        for el in history:
            mask = el.mask_property
            segmentation = SegmentationProfile(
                name="Unknown",
                algorithm=el.segmentation_parameters["algorithm_name"],
                values=el.segmentation_parameters["values"],
            )
            new_el = SegmentationPipelineElement(mask_property=mask, segmentation=segmentation)
            mask_history.append(new_el)
        name = self._settings.last_executed_algorithm
        if not name:
            QMessageBox.information(self, "No segmentation", "No segmentation executed", QMessageBox.Ok)
            return
        values = self._settings.get(f"algorithms.{name}", {})
        if len(values) == 0:
            QMessageBox.information(self, "Some problem", "Pleas run execution again", QMessageBox.Ok)
            return
        current_segmentation = SegmentationProfile(name="Unknown", algorithm=name, values=values)

        while True:
            text, ok = QInputDialog.getText(self, "Pipeline name", "Input pipeline name here")
            if not ok:
                return
            if text in self._settings.segmentation_pipelines:
                if QMessageBox.No == QMessageBox.warning(
                    self,
                    "Already exists",
                    "Profile with this name already exist. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                ):
                    continue
            profile = SegmentationPipeline(name=text, segmentation=current_segmentation, mask_history=mask_history)
            self._settings.segmentation_pipelines[text] = profile
            self._settings.dump()
            self.choose_pipe.addItem(text)
            break

    def choose_pipeline(self, text):
        if text == "<none>":
            return
        pipeline = self._settings.segmentation_pipelines[text]
        process_thread = CalculatePipelineThread(self._settings.image, self._settings.mask, pipeline)
        dial = WaitingDialog(process_thread)

        if dial.exec() and process_thread.result:
            pipeline_result = process_thread.result
            self._settings.mask = pipeline_result.mask
            self._settings.segmentation = pipeline_result.segmentation
            self._settings.full_segmentation = pipeline_result.full_segmentation
            self._settings.set_history(pipeline_result.history)
            self.label.setText(pipeline_result.description)
            self.algorithm_choose_widget.change_algorithm(pipeline.segmentation.algorithm, pipeline.segmentation.values)
        self.choose_pipe.setCurrentIndex(0)

    def update_tooltips(self):
        for i in range(1, self.choose_profile.count()):
            if self.choose_profile.itemData(i, Qt.ToolTipRole) is not None:
                continue
            text = self.choose_profile.itemText(i)
            profile: SegmentationProfile = self._settings.segmentation_profiles[text]
            tool_tip_text = str(profile)
            self.choose_profile.setItemData(i, tool_tip_text, Qt.ToolTipRole)
        for i in range(1, self.choose_pipe.count()):
            if self.choose_pipe.itemData(i, Qt.ToolTipRole) is not None:
                continue
            text = self.choose_pipe.itemText(i)
            profile: SegmentationPipeline = self._settings.segmentation_pipelines[text]
            tool_tip_text = str(profile)
            self.choose_pipe.setItemData(i, tool_tip_text, Qt.ToolTipRole)

    @staticmethod
    def update_combo_box(combo_box: QComboBox, dkt: dict):
        current_names = set(dkt.keys())
        prev_names = set([combo_box.itemText(i) for i in range(1, combo_box.count())])
        new_names = current_names - prev_names
        delete_names = prev_names - current_names
        if len(delete_names) > 0:
            i = 1
            while i < combo_box.count():
                if combo_box.itemText(i) in delete_names:
                    combo_box.removeItem(i)
                else:
                    i += 1
        if len(new_names) > 0:
            combo_box.addItems(list(sorted(new_names)))

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            # update combobox for segmentation
            self.update_combo_box(self.choose_profile, self._settings.segmentation_profiles)
            # update combobox for pipeline
            self.update_combo_box(self.choose_pipe, self._settings.segmentation_pipelines)
            self.update_tooltips()
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return) and (event.modifiers() == Qt.ControlModifier):
            self.execute_btn.click()

    def save_profile(self):
        widget: InteractiveAlgorithmSettingsWidget = self.algorithm_choose_widget.current_widget()
        while True:
            text, ok = QInputDialog.getText(self, "Profile Name", "Input profile name here")
            if not ok:
                return
            if text in self._settings.segmentation_profiles:
                if QMessageBox.No == QMessageBox.warning(
                    self,
                    "Already exists",
                    "Profile with this name already exist. Overwrite?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                ):
                    continue
            resp = SegmentationProfile(text, widget.name, widget.get_values())
            self._settings.segmentation_profiles[text] = resp
            self._settings.dump()
            self.choose_profile.addItem(text)
            self.update_tooltips()
            break

    def change_profile(self, val):
        self.choose_profile.setToolTip("")
        if val == "<none>":
            return
        interactive = self.interactive_use.isChecked()
        self.interactive_use.setChecked(False)
        profile = self._settings.segmentation_profiles[val]
        self.algorithm_choose_widget.change_algorithm(profile.algorithm, profile.values)
        self.choose_profile.blockSignals(True)
        self.choose_profile.setCurrentIndex(0)
        self.choose_profile.blockSignals(False)
        self.interactive_use.setChecked(interactive)

    @property
    def segmentation(self):
        return self._settings.segmentation

    @property
    def interactive(self):
        return self.interactive_use.isChecked()

    def hide_left_panel(self, val):
        self._settings.set_in_profile("hide_left_panel", val)
        if val:
            self.synchronize_val = self.synchronize_checkbox.isChecked()
            self.synchronize_checkbox.setChecked(False)
        else:
            self.synchronize_checkbox.setChecked(self.synchronize_val)
        self.synchronize_checkbox.setDisabled(val)
        self.left_panel.parent().setHidden(val)

    def interactive_change(self, val):
        if val:
            self.execute_algorithm()

    def algorithm_change(self, val):
        self._settings.set("current_algorithm", val)
        if self.interactive:
            self.execute_algorithm()

    def interactive_algorithm_execute(self):
        if self.interactive:
            self.execute_algorithm()

    def execute_algorithm(self):
        widget: InteractiveAlgorithmSettingsWidget = self.algorithm_choose_widget.current_widget()
        if self._settings.image.is_time and not widget.algorithm.support_time():
            QMessageBox.information(
                self, "Not supported", "This algorithm do not support time data. " "You can convert it in image adjust"
            )
            return
        if self._settings.image.is_stack and not widget.algorithm.support_z():
            QMessageBox.information(
                self, "Not supported", "This algorithm do not support stack data. " "You can convert it in image adjust"
            )
            return
        self._settings.last_executed_algorithm = widget.name
        self.execute_btn.setDisabled(True)
        self.interactive_use.setDisabled(True)
        widget.execute()

    def execution_done(self, segmentation: SegmentationResult):
        if segmentation.info_text != "":
            QMessageBox.information(self, "Algorithm info", segmentation.info_text)
        self._settings.segmentation = segmentation.segmentation
        self.compare_btn.setEnabled(
            isinstance(segmentation.segmentation, np.ndarray) and np.any(segmentation.segmentation)
        )
        self._settings.noise_remove_image_part = segmentation.cleaned_channel
        self._settings.full_segmentation = segmentation.full_segmentation
        self.label.setText(self.sender().get_info_text())

    def showEvent(self, _event):
        self.hide_left_panel_chk.setChecked(self._settings.get_from_profile("hide_left_panel", False))


class MainMenu(BaseMainMenu):
    def __init__(self, settings: PartSettings, main_window):
        super().__init__(settings, main_window)
        self.open_btn = QPushButton("Open")
        self.save_btn = QPushButton("Save")
        self.advanced_btn = QPushButton("Settings and Measurement")
        self.mask_manager_btn = QPushButton("Mask manager")
        self.batch_processing_btn = QPushButton("Batch Processing")

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.advanced_btn)
        layout.addWidget(self.mask_manager_btn)
        layout.addWidget(self.batch_processing_btn)
        self.setLayout(layout)

        self.open_btn.clicked.connect(self.load_data)
        self.save_btn.clicked.connect(self.save_file)
        self.advanced_btn.clicked.connect(self.advanced_window_show)
        self.mask_manager_btn.clicked.connect(self.mask_manager)
        self.batch_processing_btn.clicked.connect(self.batch_window)
        self.setFocusPolicy(Qt.StrongFocus)
        # self.test_btn.clicked.connect(self.test_fun)

    def resizeEvent(self, event: QResizeEvent):
        if event.size().width() < 800:
            self.batch_processing_btn.hide()
        else:
            self.batch_processing_btn.show()

    def keyPressEvent(self, event: QKeyEvent):
        if event.matches(QKeySequence.Save):
            self.save_file()
        elif event.matches(QKeySequence.Open):
            self.load_data()
        super().keyPressEvent(event)

    def save_file(self):
        base_values = self.settings.get("save_parameters", dict())
        dial = SaveDialog(
            save_dict, system_widget=False, base_values=base_values, history=self.settings.get_path_history()
        )
        dial.selectFile(os.path.splitext(os.path.basename(self.settings.image_path))[0])
        dial.setDirectory(
            self.settings.get("io.save_directory", self.settings.get("io.open_directory", str(Path.home())))
        )
        dial.selectNameFilter(self.settings.get("io.save_filter", ""))
        if dial.exec():
            save_location, selected_filter, save_class, values = dial.get_result()
            project_info = self.settings.get_project_info()
            self.settings.set("io.save_filter", selected_filter)
            save_dir = os.path.dirname(save_location)
            self.settings.set("io.save_directory", save_dir)
            self.settings.add_path_history(save_dir)
            base_values[selected_filter] = values

            def exception_hook(exception):
                from qtpy.QtWidgets import QApplication
                from qtpy.QtCore import QMetaObject

                instance = QApplication.instance()
                if isinstance(exception, ValueError):
                    instance.warning = "Save error", f"Error during saving\n{exception}"
                    QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
                else:
                    raise exception

            dial2 = ExecuteFunctionDialog(
                save_class.save, [save_location, project_info, values], exception_hook=exception_hook
            )
            dial2.exec()

    def mask_manager(self):
        if self.settings.segmentation is None:
            QMessageBox.information(self, "No segmentation", "Cannot create mask without segmentation")
            return
        dial = MaskDialog(self.settings)
        dial.exec_()

    def load_data(self):
        def exception_hook(exception):
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QMetaObject

            instance = QApplication.instance()
            if isinstance(exception, ValueError) and exception.args[0] == "Incompatible shape of mask and image":
                instance.warning = (
                    "Open error",
                    "Most probably you try to load mask from other image. " "Check selected files",
                )
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, MemoryError):
                instance.warning = "Open error", f"Not enough memory to read this image: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, IOError):
                instance.warning = "Open error", f"Some problem with reading from disc: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, KeyError):
                instance.warning = "Open error", f"Some problem project file: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
                print(exception, file=sys.stderr)
            elif isinstance(exception, WrongFileTypeException):
                instance.warning = (
                    "Open error",
                    f"No needed files inside archive. " f"Most probably you choose file from segmentation mask",
                )
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            else:
                raise exception

        try:
            dial = CustomLoadDialog(load_functions.load_dict, history=self.settings.get_path_history())
            dial.setDirectory(self.settings.get("io.open_directory", str(Path.home())))
            dial.selectNameFilter(self.settings.get("io.open_filter", next(iter(load_functions.load_dict.keys()))))
            if dial.exec_():
                result = dial.get_result()
                self.settings.set("io.open_filter", result.selected_filter)
                load_dir = os.path.dirname(result.load_location[0])
                self.settings.set("io.open_directory", load_dir)
                self.settings.add_path_history(load_dir)
                dial2 = ExecuteFunctionDialog(
                    result.load_class.load,
                    [result.load_location],
                    {"metadata": {"default_spacing": self.settings.image_spacing}},
                    exception_hook=exception_hook,
                )
                if dial2.exec():
                    result = dial2.get_result()
                    self.set_data(result)

        except ValueError as e:
            QMessageBox.warning(self, "Open error", "{}".format(e))

    def batch_window(self):
        if self.main_window.batch_window is not None:
            if self.main_window.batch_window.isVisible():
                self.main_window.batch_window.activateWindow()
            else:
                self.main_window.batch_window.show()
        else:
            self.main_window.batch_window = BatchWindow(self.settings)
            self.main_window.batch_window.show()

    def advanced_window_show(self):
        if self.main_window.advanced_window.isVisible():
            self.main_window.advanced_window.activateWindow()
        else:
            self.main_window.advanced_window.show()


class MaskDialog(MaskDialogBase):
    def next_mask(self):
        project_info: ProjectTuple = self.settings.get_project_info()
        mask_property = self.mask_widget.get_mask_property()
        self.settings.set("mask_manager.mask_property", mask_property)
        mask = calculate_mask(mask_property, project_info.segmentation, project_info.mask, project_info.image.spacing)
        self.settings.add_history_element(create_history_element_from_project(project_info, mask_property,))
        if self.settings.history_redo_size():
            history: HistoryElement = self.settings.history_next_element()
            self.settings.set("current_algorithm", history.segmentation_parameters["algorithm_name"])
            self.settings.set(
                f"algorithm.{history.segmentation_parameters['algorithm_name']}",
                history.segmentation_parameters["values"],
            )
        self.settings.mask = mask
        self.close()

    def prev_mask(self):
        history: HistoryElement = self.settings.history_pop()
        algorithm_name = self.settings.last_executed_algorithm
        algorithm_values = self.settings.get(f"algorithms.{algorithm_name}")
        self.settings.fix_history(algorithm_name=algorithm_name, algorithm_values=algorithm_values)
        self.settings.set("current_algorithm", history.segmentation_parameters["algorithm_name"])
        self.settings.set(
            f"algorithm.{history.segmentation_parameters['algorithm_name']}", history.segmentation_parameters["values"]
        )
        history.arrays.seek(0)
        seg = np.load(history.arrays)
        history.arrays.seek(0)
        self.settings.segmentation = seg["segmentation"]
        self.settings.full_segmentation = seg["full_segmentation"]
        if "mask" in seg:
            self.settings.mask = seg["mask"]
        else:
            self.settings.mask = None
        self.close()


class MainWindow(BaseMainWindow):
    @classmethod
    def get_setting_class(cls) -> Type[PartSettings]:
        return PartSettings

    initial_image_path = PartSegData.segmentation_analysis_default_image

    def __init__(
        self, config_folder=CONFIG_FOLDER, title="PartSeg", settings=None, signal_fun=None, initial_image=None
    ):
        super().__init__(config_folder, title, settings, signal_fun)
        self.files_num = 2
        self.setMinimumWidth(600)
        self.main_menu = MainMenu(self.settings, self)
        # self.channel_control1 = ChannelControl(self.settings, name="raw_control", text="Left panel:")
        self.channel_control2 = ChannelProperty(self.settings, start_name="result_control")
        self.raw_image = CompareImageView(self.settings, self.channel_control2, "raw_image")
        self.measurements = MeasurementWidget(self.settings)
        self.left_stack = StackedWidgetWithSelector()
        self.left_stack.addWidget(self.raw_image, "Image")
        self.left_stack.addWidget(self.measurements, "Measurements")
        self.result_image = ResultImageView(self.settings, self.channel_control2, "result_image")
        self.color_bar = ColorBar(self.settings, [self.raw_image, self.result_image])
        self.info_text = QLabel()
        self.info_text.setMinimumHeight(25)
        self.raw_image.text_info_change.connect(self.info_text.setText)
        self.result_image.text_info_change.connect(self.info_text.setText)
        self.synchronize_tool = SynchronizeView(self.raw_image, self.result_image, self)
        # image_view_control = self.image_view.get_control_view()
        self.options_panel = Options(
            self.settings, self.channel_control2, self.raw_image, self.result_image, self.synchronize_tool
        )
        # self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)
        self.advanced_window = SegAdvancedWindow(self.settings, reload_list=[self.reload])
        self.batch_window = None  # BatchWindow(self.settings)

        self.multiple_files = MultipleFileWidget(self.settings, load_functions.load_dict, True)

        if initial_image is None:
            reader = TiffImageReader()
            im = reader.read(self.initial_image_path)
            im.file_path = ""
            self.settings.image = im
        else:
            self.settings.image = initial_image

        icon = QIcon(os.path.join(PartSegData.icons_dir, "icon.png"))
        self.setWindowIcon(icon)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("&Open").triggered.connect(self.main_menu.load_data)
        file_menu.addAction("&Save").triggered.connect(self.main_menu.save_file)
        file_menu.addAction("Batch processing").triggered.connect(self.main_menu.batch_window)
        image_menu = menu_bar.addMenu("Image operations")
        image_menu.addAction("Image adjustment").triggered.connect(self.image_adjust_exec)
        image_menu.addAction("Mask manager").triggered.connect(self.main_menu.mask_manager)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("State directory").triggered.connect(self.show_settings_directory)
        help_menu.addAction("About").triggered.connect(self.show_about_dialog)

        layout = QGridLayout()
        # layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.left_stack.selector)
        info_layout.addWidget(self.options_panel.compare_btn)
        info_layout.addWidget(self.info_text, 1, Qt.AlignHCenter)

        image_layout =  EqualColumnLayout()
        image_layout.addWidget(self.left_stack)
        image_layout.addWidget(self.result_image)

        layout.setSpacing(0)
        layout.addWidget(self.main_menu, 0, 0, 1, 3)
        layout.addLayout(info_layout, 1, 1, 1, 2)
        layout.addWidget(self.multiple_files, 2, 0)
        layout.addWidget(self.color_bar, 2, 1)
        #layout.addWidget(self.left_stack, 2, 2)
        #layout.addWidget(self.result_image, 2, 3)
        layout.addLayout(image_layout, 2, 2, 1, 1)
        layout.addWidget(self.options_panel, 0, 3, 3, 1)
        layout.setColumnStretch(2, 1)
        #layout.setColumnStretch(3, 1)
        widget = QWidget()
        widget.setLayout(layout)
        # self.multiple_files.setHidden(True)
        self.setCentralWidget(widget)
        try:
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))
        except KeyError:
            pass

    def image_read(self):
        self.raw_image.set_image()
        self.raw_image.reset_image_size()
        self.result_image.set_image()
        self.result_image.reset_image_size()
        self.options_panel.interactive_algorithm_execute()
        self.setWindowTitle(f"{self.title_base}: {os.path.basename(self.settings.image_path)}")

    def read_drop(self, paths):
        self._read_drop(paths, load_functions)

    def reload(self):
        self.options_panel.algorithm_choose_widget.reload(algorithm_description.analysis_algorithm_dict)

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            self.multiple_files.setVisible(self.settings.get("multiple_files", False))
        return super().event(event)

    def closeEvent(self, event):
        # print(self.settings.dump_view_profiles())
        # print(self.settings.segmentation_dict["default"].my_dict)
        self.settings.set_in_profile("main_window_geometry", self.saveGeometry().toHex().data().decode("ascii"))
        self.options_panel.algorithm_choose_widget.recursive_get_values()
        if self.batch_window is not None:
            if self.batch_window.is_working():
                ret = QMessageBox.warning(
                    self,
                    "Batch work",
                    "Batch work is not finished. " "Would you like to terminate it?",
                    QMessageBox.No | QMessageBox.Yes,
                )
                if ret == QMessageBox.Yes:
                    self.batch_window.terminate()
                else:
                    event.ignore()
                    return
            self.batch_window.close()
        self.advanced_window.close()
        self.settings.dump()
        del self.batch_window
        del self.advanced_window

    @staticmethod
    def get_project_info(file_path, image):
        return ProjectTuple(file_path=file_path, image=image)

    def set_data(self, data):
        self.main_menu.set_data(data)
