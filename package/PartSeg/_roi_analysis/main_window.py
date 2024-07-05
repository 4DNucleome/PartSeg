import os
from contextlib import suppress
from typing import Type

from qtpy.QtCore import QByteArray, Qt
from qtpy.QtGui import QIcon, QKeyEvent, QKeySequence, QResizeEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from superqt import ensure_main_thread

import PartSegData
from PartSeg import state_store
from PartSeg._roi_analysis.advanced_window import SegAdvancedWindow
from PartSeg._roi_analysis.batch_window import BatchWindow
from PartSeg._roi_analysis.calculation_pipeline_thread import CalculatePipelineThread
from PartSeg._roi_analysis.image_view import CompareImageView, ResultImageView, SynchronizeView
from PartSeg._roi_analysis.measurement_widget import MeasurementWidget
from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg.common_backend.base_settings import IO_SAVE_DIRECTORY
from PartSeg.common_backend.except_hook import show_warning
from PartSeg.common_gui.algorithms_description import AlgorithmChoose, InteractiveAlgorithmSettingsWidget
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.custom_load_dialog import PLoadDialog
from PartSeg.common_gui.custom_save_dialog import PSaveDialog
from PartSeg.common_gui.equal_column_layout import EqualColumnLayout
from PartSeg.common_gui.exception_hooks import OPEN_ERROR, load_data_exception_hook
from PartSeg.common_gui.main_window import OPEN_DIRECTORY, OPEN_FILE, OPEN_FILE_FILTER, BaseMainMenu, BaseMainWindow
from PartSeg.common_gui.mask_widget import MaskDialogBase
from PartSeg.common_gui.multiple_file_widget import MultipleFileWidget
from PartSeg.common_gui.searchable_combo_box import SearchComboBox
from PartSeg.common_gui.stack_image_view import ColorBar
from PartSeg.common_gui.stacked_widget_with_selector import StackedWidgetWithSelector
from PartSeg.common_gui.universal_gui_part import TextShow
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog, WaitingDialog
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import ProjectTuple, algorithm_description, load_functions
from PartSegCore.analysis.analysis_utils import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.analysis.io_utils import create_history_element_from_project
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore.io_utils import WrongFileTypeException
from PartSegCore.project_info import HistoryElement, calculate_mask_from_project
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.algorithm_base import ROIExtractionResult
from PartSegCore.utils import EventedDict
from PartSegImage import TiffImageReader

CONFIG_FOLDER = os.path.join(state_store.save_folder, "analysis")


class Options(QWidget):
    def __init__(
        self,
        settings: PartSettings,
        channel_control2: ChannelProperty,
        left_image: ResultImageView,
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
        self.choose_pipe = SearchComboBox()
        self.choose_pipe.addItem("<none>")
        self.choose_pipe.textActivated.connect(self.choose_pipeline)
        self.choose_pipe.setToolTip("Execute chosen pipeline")
        self.save_profile_btn = QPushButton("Save profile")
        self.save_profile_btn.setToolTip("Save values from current view")
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.choose_profile = SearchComboBox()
        self.choose_profile.addItem("<none>")
        self.choose_profile.setToolTip("Select profile to restore its settings. Execute if interactive is checked")
        # image state
        self.compare_btn = QPushButton("Compare")
        self.compare_btn.setDisabled(True)
        self.compare_btn.clicked.connect(self.compare_action)
        left_image.hide_signal.connect(self.compare_btn.setHidden)

        self.update_tooltips()
        self.choose_profile.textActivated.connect(self.change_profile)
        self.interactive_use.stateChanged.connect(self.execute_btn.setDisabled)
        self.interactive_use.stateChanged.connect(self.interactive_change)
        self.algorithm_choose_widget = AlgorithmChoose(settings, algorithm_description.AnalysisAlgorithmSelection)
        self.algorithm_choose_widget.result.connect(self.execution_done)
        self.algorithm_choose_widget.finished.connect(self.calculation_finished)
        self.algorithm_choose_widget.value_changed.connect(self.interactive_algorithm_execute)
        self.algorithm_choose_widget.algorithm_changed.connect(self.interactive_algorithm_execute)
        self._settings.roi_profiles_changed.connect(self._update_profiles)
        self._settings.roi_pipelines_changed.connect(self._update_pipelines)
        self._update_pipelines()
        self._update_profiles()

        self.label = TextShow()

        self.setup_ui()

        settings.roi_changed.connect(self._refresh_compare_btn)
        settings.image_changed.connect(self._reset_compare_btn)

    def setup_ui(self):
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
        layout.addWidget(self.label)
        layout2.addWidget(self.hide_left_panel_chk)
        layout2.addWidget(self.synchronize_checkbox)
        layout.addLayout(layout2)
        layout.addWidget(self._ch_control2)
        self.setLayout(layout)

    @ensure_main_thread
    def _update_profiles(self):
        self.update_combo_box(self.choose_profile, self._settings.roi_profiles)
        self.update_tooltips()

    @ensure_main_thread
    def _update_pipelines(self):
        self.update_combo_box(self.choose_pipe, self._settings.roi_pipelines)
        self.update_tooltips()

    def compare_action(self):
        if self.compare_btn.text() == "Compare":
            self._settings.set_segmentation_to_compare(self._settings.roi_info)
            self.compare_btn.setText("Remove")
        else:
            self._settings.set_segmentation_to_compare(ROIInfo(None))
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
            segmentation = ROIExtractionProfile(
                name="Unknown",
                algorithm=el.roi_extraction_parameters["algorithm_name"],
                values=el.roi_extraction_parameters["values"],
            )
            new_el = SegmentationPipelineElement(mask_property=mask, segmentation=segmentation)
            mask_history.append(new_el)
        name = self._settings.last_executed_algorithm
        if not name:
            QMessageBox.information(self, "No segmentation", "No segmentation executed", QMessageBox.Ok)
            return
        values = self._settings.get_algorithm(f"algorithms.{name}", {})
        if isinstance(values, (dict, EventedDict)) and len(values) == 0:
            QMessageBox.information(self, "Some problem", "Please run execution again", QMessageBox.Ok)
            return
        current_segmentation = ROIExtractionProfile(name="Unknown", algorithm=name, values=values)

        while True:
            text, ok = QInputDialog.getText(self, "Pipeline name", "Input pipeline name here")
            if not ok:
                return
            if text in self._settings.roi_pipelines and QMessageBox.No == QMessageBox.warning(
                self,
                "Already exists",
                "Profile with this name already exist. Overwrite?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ):
                continue
            profile = SegmentationPipeline(name=text, segmentation=current_segmentation, mask_history=mask_history)
            self._settings.roi_pipelines[text] = profile
            self._settings.dump()
            break

    def choose_pipeline(self, text):
        if text == "<none>":
            return
        pipeline = self._settings.roi_pipelines[text]
        process_thread = CalculatePipelineThread(self._settings.image, self._settings.mask, pipeline)
        dial = WaitingDialog(process_thread)

        if dial.exec_() and process_thread.result:
            pipeline_result = process_thread.result
            self._settings.mask = pipeline_result.mask
            self._settings.roi = pipeline_result.roi_info.roi
            self._settings.set_history(pipeline_result.history)
            self.label.setText(pipeline_result.description)
            self.algorithm_choose_widget.change_algorithm(pipeline.segmentation.algorithm, pipeline.segmentation.values)
        self.choose_pipe.setCurrentIndex(0)

    def update_tooltips(self):
        for i in range(1, self.choose_profile.count()):
            if self.choose_profile.itemData(i, Qt.ItemDataRole.ToolTipRole) is not None:
                continue
            text = self.choose_profile.itemText(i)
            profile: ROIExtractionProfile = self._settings.roi_profiles[text]
            tool_tip_text = str(profile)
            self.choose_profile.setItemData(i, tool_tip_text, Qt.ItemDataRole.ToolTipRole)
        for i in range(1, self.choose_pipe.count()):
            if self.choose_pipe.itemData(i, Qt.ItemDataRole.ToolTipRole) is not None:
                continue
            text = self.choose_pipe.itemText(i)
            profile: SegmentationPipeline = self._settings.roi_pipelines[text]
            tool_tip_text = str(profile)
            self.choose_pipe.setItemData(i, tool_tip_text, Qt.ItemDataRole.ToolTipRole)

    @staticmethod
    def update_combo_box(combo_box: QComboBox, dkt: dict):
        current_names = set(dkt.keys())
        prev_names = {combo_box.itemText(i) for i in range(1, combo_box.count())}
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
            combo_box.addItems(sorted(new_names))

    def keyPressEvent(self, event: QKeyEvent):
        if (
            event.key() in [Qt.Key.Key_Enter, Qt.Key.Key_Return]
            and event.modifiers() == Qt.KeyboardModifier.ControlModifier
        ):
            self.execute_btn.click()

    def save_profile(self):
        widget: InteractiveAlgorithmSettingsWidget = self.algorithm_choose_widget.current_widget()
        while True:
            text, ok = QInputDialog.getText(self, "Profile Name", "Input profile name here")
            if not ok:
                return
            if text in self._settings.roi_profiles and QMessageBox.StandardButton.No == QMessageBox.warning(
                self,
                "Already exists",
                "Profile with this name already exist. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            ):
                continue
            resp = ROIExtractionProfile(name=text, algorithm=widget.name, values=widget.get_values())
            self._settings.roi_profiles[text] = resp
            self._settings.dump()
            break

    def change_profile(self, val):
        self.choose_profile.setToolTip("")
        if val == "<none>":
            return
        interactive = self.interactive_use.isChecked()
        self.interactive_use.setChecked(False)
        profile = self._settings.roi_profiles[val]
        self.algorithm_choose_widget.change_algorithm(profile.algorithm, profile.values)
        self.choose_profile.blockSignals(True)
        self.choose_profile.setCurrentIndex(0)
        self.choose_profile.blockSignals(False)
        self.interactive_use.setChecked(interactive)

    @property
    def segmentation(self):
        return self._settings.roi

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
                self, "Not supported", "This algorithm do not support time data. You can convert it in image adjust"
            )
            return
        if self._settings.image.is_stack and not widget.algorithm.support_z():
            QMessageBox.information(
                self, "Not supported", "This algorithm do not support stack data. You can convert it in image adjust"
            )
            return
        self._settings.last_executed_algorithm = widget.name
        self.execute_btn.setDisabled(True)
        self.interactive_use.setDisabled(True)
        widget.execute()

    def execution_done(self, segmentation: ROIExtractionResult):
        if segmentation.info_text:
            QMessageBox.information(self, "Algorithm info", segmentation.info_text)
        self._settings.set_segmentation_result(segmentation)
        self.label.setText(self.sender().get_info_text())

    def _refresh_compare_btn(self):
        self.compare_btn.setEnabled(bool(self._settings.roi_info.bound_info))

    def _reset_compare_btn(self):
        self.compare_btn.setText("Compare")

    def showEvent(self, _event):
        self.hide_left_panel_chk.setChecked(self._settings.get_from_profile("hide_left_panel", False))


class MainMenu(BaseMainMenu):
    def __init__(self, settings: PartSettings, main_window):
        super().__init__(settings, main_window)
        self.settings = settings
        self.open_btn = QPushButton("Open")
        self.save_btn = QPushButton("Save")
        self.advanced_btn = QPushButton("Settings and Measurement")
        self.mask_manager_btn = QPushButton("Mask manager")
        self.batch_processing_btn = QPushButton("Batch Processing")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 4, 4)
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
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def resizeEvent(self, event: QResizeEvent):
        if event.size().width() < 800:
            self.batch_processing_btn.setVisible(False)
        else:
            self.batch_processing_btn.setVisible(True)

    def keyPressEvent(self, event: QKeyEvent):
        if event.matches(QKeySequence.Save):
            self.save_file()
        elif event.matches(QKeySequence.Open):
            self.load_data()
        super().keyPressEvent(event)

    def save_file(self):
        base_values = self.settings.get("save_parameters", {})
        dial = PSaveDialog(
            save_dict,
            system_widget=False,
            base_values=base_values,
            settings=self.settings,
            path=IO_SAVE_DIRECTORY,
            filter_path="io.save_filter",
        )
        dial.selectFile(os.path.splitext(os.path.basename(self.settings.image_path))[0])
        if dial.exec_():
            save_location, selected_filter, save_class, values = dial.get_result()
            project_info = self.settings.get_project_info()
            if save_class.need_segmentation() and project_info.roi_info.roi is None:
                QMessageBox.information(self, "No segmentation", "Cannot save without segmentation")
                return
            if save_class.need_mask() and project_info.mask is None:
                QMessageBox.information(self, "No mask", "Cannot save without mask")
                return
            base_values[selected_filter] = values

            def exception_hook(exception):  # pragma: no cover
                if isinstance(exception, ValueError):
                    show_warning("Save error", f"Error during saving\n{exception}", exception=exception)
                else:
                    raise exception

            dial2 = ExecuteFunctionDialog(
                save_class.save, [save_location, project_info, values], exception_hook=exception_hook
            )
            dial2.exec_()

    def mask_manager(self):
        if self.settings.roi is None:
            QMessageBox.information(self, "No segmentation", "Cannot create mask without segmentation")
            return
        dial = MaskDialog(self.settings)
        dial.exec_()

    def load_data(self):
        def exception_hook(exception):
            if isinstance(exception, WrongFileTypeException):
                show_warning(
                    OPEN_ERROR,
                    "No needed files inside archive. Most probably you choose file from segmentation mask",
                    exception=exception,
                )
            else:
                load_data_exception_hook(exception)

        try:
            dial = PLoadDialog(
                load_functions.load_dict, settings=self.settings, path=OPEN_DIRECTORY, filter_path=OPEN_FILE_FILTER
            )
            file_path = self.settings.get(OPEN_FILE, "")
            if os.path.isfile(file_path):
                dial.selectFile(file_path)
            if dial.exec_():
                result = dial.get_result()
                self.settings.set(OPEN_FILE, result.load_location[0])
                self.settings.add_last_files(result.load_location, result.load_class.get_name())
                dial2 = ExecuteFunctionDialog(
                    result.load_class.load,
                    [result.load_location],
                    {"metadata": {"default_spacing": self.settings.image_spacing}},
                    exception_hook=exception_hook,
                )
                if dial2.exec_():
                    result = dial2.get_result()
                    self.set_data(result)

        except ValueError as e:  # pragma: no cover
            show_warning("Open error", f"{e}", exception=e)

    def batch_window(self):
        if self.main_window.batch_window is None:
            self.main_window.batch_window = BatchWindow(self.settings)
            self.main_window.batch_window.show()

        elif self.main_window.batch_window.isVisible():
            self.main_window.batch_window.activateWindow()
        else:
            self.main_window.batch_window.show()

    def advanced_window_show(self):
        if self.main_window.advanced_window.isVisible():
            self.main_window.advanced_window.activateWindow()
        else:
            self.main_window.advanced_window.show()


class MaskDialog(MaskDialogBase):
    # FIXME add tests
    def next_mask(self):
        project_info: ProjectTuple = self.settings.get_project_info()
        mask_property = self.mask_widget.get_mask_property()
        self.settings.set("mask_manager.mask_property", mask_property)
        mask = calculate_mask_from_project(mask_description=mask_property, project=project_info)
        self.settings.add_history_element(
            create_history_element_from_project(
                project_info,
                mask_property,
            )
        )
        if self.settings.history_redo_size():
            history: HistoryElement = self.settings.history_next_element()
            self.settings.set("current_algorithm", history.roi_extraction_parameters["algorithm_name"])
            self.settings.set(
                f"algorithm.{history.roi_extraction_parameters['algorithm_name']}",
                history.roi_extraction_parameters["values"],
            )
        self.settings.mask = mask
        self.close()

    def prev_mask(self):
        history: HistoryElement = self.settings.history_pop()
        algorithm_name = self.settings.last_executed_algorithm
        algorithm_values = self.settings.get_algorithm(f"algorithms.{algorithm_name}")
        self.settings.fix_history(algorithm_name=algorithm_name, algorithm_values=algorithm_values)
        self.settings.set("current_algorithm", history.roi_extraction_parameters["algorithm_name"])
        self.settings.set(
            f"algorithm.{history.roi_extraction_parameters['algorithm_name']}",
            history.roi_extraction_parameters["values"],
        )
        self.settings.roi, self.settings.mask = history.get_roi_info_and_mask()
        self.close()


class MainWindow(BaseMainWindow):
    settings: PartSettings

    @classmethod
    def get_setting_class(cls) -> Type[PartSettings]:
        return PartSettings

    initial_image_path = PartSegData.segmentation_analysis_default_image

    def __init__(  # noqa: PLR0915
        self, config_folder=CONFIG_FOLDER, title="PartSeg", settings=None, signal_fun=None, initial_image=None
    ):
        super().__init__(config_folder, title, settings, load_functions.load_dict, signal_fun)
        self.channel_info = "result_image"
        self.files_num = 2
        self.setMinimumWidth(600)
        # this isinstance is only for type hinting in IDE
        self.main_menu = MainMenu(self.settings, self)
        self.channel_control2 = ChannelProperty(self.settings, start_name="result_image")
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
        self.options_panel = Options(self.settings, self.channel_control2, self.raw_image, self.synchronize_tool)
        self.settings.image_changed.connect(self.image_read)
        self.advanced_window = SegAdvancedWindow(self.settings, reload_list=[self.reload])
        self.batch_window = None  # BatchWindow(self.settings)

        self.multiple_files = MultipleFileWidget(self.settings, load_functions.load_dict, True)
        self.multiple_files.setVisible(self.settings.get("multiple_files_widget", False))

        if initial_image is None:
            reader = TiffImageReader()
            im = reader.read(self.initial_image_path)
            im.file_path = ""
            self.settings.image = im
        elif initial_image is not False:
            self.settings.image = initial_image

        self._setup_menu_bar()

        icon = QIcon(os.path.join(PartSegData.icons_dir, "icon.png"))
        self.setWindowIcon(icon)

        layout = QGridLayout()
        layout.setSpacing(0)
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.left_stack.selector)
        info_layout.addWidget(self.options_panel.compare_btn)
        info_layout.addWidget(self.info_text, 1, Qt.AlignmentFlag.AlignHCenter)

        image_layout = EqualColumnLayout()
        image_layout.addWidget(self.left_stack)
        image_layout.addWidget(self.result_image)

        layout.setSpacing(0)
        layout.addWidget(self.main_menu, 0, 0, 1, 3)
        layout.addLayout(info_layout, 3, 1, 1, 2)
        layout.addWidget(self.multiple_files, 2, 0)
        layout.addWidget(self.color_bar, 2, 1)
        layout.addLayout(image_layout, 2, 2, 1, 1)
        layout.addWidget(self.options_panel, 0, 3, 3, 1)
        layout.setColumnStretch(2, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        with suppress(KeyError):
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))

    def _setup_menu_bar(self):
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("&Open").triggered.connect(self.main_menu.load_data)
        file_menu.addMenu(self.recent_file_menu)
        file_menu.addAction("&Save").triggered.connect(self.main_menu.save_file)
        file_menu.addAction("Batch processing").triggered.connect(self.main_menu.batch_window)
        view_menu = menu_bar.addMenu("View")
        view_menu.addAction("Settings and Measurement").triggered.connect(self.main_menu.advanced_window_show)
        view_menu.addAction("Additional output").triggered.connect(self.additional_layers_show)
        view_menu.addAction("Additional output with data").triggered.connect(lambda: self.additional_layers_show(True))
        view_menu.addAction("Napari viewer").triggered.connect(self.napari_viewer_show)
        view_menu.addAction("Toggle Multiple Files").triggered.connect(self.toggle_multiple_files)
        view_menu.addAction("Toggle left panel").triggered.connect(self.toggle_left_panel)
        view_menu.addAction("Toggle console").triggered.connect(self._toggle_console)
        view_menu.addAction("Toggle scale bar").triggered.connect(self._toggle_scale_bar)
        action = view_menu.addAction("Screenshot right panel")
        action.triggered.connect(self.screenshot(self.result_image))
        action.setShortcut(QKeySequence.Print)
        view_menu.addAction("Screenshot left panel").triggered.connect(self.screenshot(self.raw_image))
        image_menu = menu_bar.addMenu("Image operations")
        image_menu.addAction("Image adjustment").triggered.connect(self.image_adjust_exec)
        image_menu.addAction("Mask manager").triggered.connect(self.main_menu.mask_manager)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("State directory").triggered.connect(self.show_settings_directory)
        help_menu.addAction("About").triggered.connect(self.show_about_dialog)

    def _toggle_scale_bar(self):
        self.raw_image.toggle_scale_bar()
        self.result_image.toggle_scale_bar()
        super()._toggle_scale_bar()

    def toggle_left_panel(self):
        self.options_panel.hide_left_panel(not self.settings.get_from_profile("hide_left_panel"))

    def image_read(self):
        super().image_read()
        self.options_panel.interactive_algorithm_execute()

    def reload(self):
        self.options_panel.algorithm_choose_widget.reload(algorithm_description.AnalysisAlgorithmSelection)

    def closeEvent(self, event):
        self.settings.set_in_profile("main_window_geometry", self.saveGeometry().toHex().data().decode("ascii"))
        self.options_panel.algorithm_choose_widget.recursive_get_values()
        if self.batch_window is not None:
            if self.batch_window.is_working():
                ret = QMessageBox.warning(
                    self,
                    "Batch work",
                    "Batch work is not finished. Would you like to terminate it?",
                    QMessageBox.No | QMessageBox.Yes,
                )
                if ret == QMessageBox.Yes:
                    self.batch_window.terminate()
                else:
                    event.ignore()
                    return
            self.batch_window.close()
        self.advanced_window.close()
        del self.batch_window
        del self.advanced_window
        super().closeEvent(event)

    @staticmethod
    def get_project_info(file_path, image, roi_info=None):
        if roi_info is None:
            roi_info = ROIInfo(None)
        return ProjectTuple(file_path=file_path, image=image, roi_info=roi_info)

    def set_data(self, data):
        self.main_menu.set_data(data)

    def change_theme(self, event):
        self.raw_image.set_theme(self.settings.theme_name)
        self.result_image.set_theme(self.settings.theme_name)
        super().change_theme(event)
