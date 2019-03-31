import os
import sys
from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, QByteArray, QEvent
from qtpy.QtGui import QIcon, QKeyEvent, QKeySequence, QResizeEvent
from qtpy.QtWidgets import QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QMessageBox, QCheckBox, QComboBox, QInputDialog, QDialog

from PartSeg.common_gui.about_dialog import AboutDialog
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.image_adjustment import ImageAdjustmentDialog
from PartSeg.common_gui.show_directory_dialog import DirectoryDialog
from PartSeg.utils.analysis import ProjectTuple
from PartSeg.utils.analysis.load_functions import load_dict
from ..common_gui.algorithms_description import InteractiveAlgorithmSettingsWidget, AlgorithmChoose
from ..common_gui.channel_control import ChannelControl
from ..common_gui.mask_widget import MaskWidget
from ..common_gui.stack_image_view import ColorBar
from ..common_gui.waiting_dialog import WaitingDialog, ExecuteFunctionDialog
from ..common_gui.multiple_file_widget import MultipleFileWidget
from ..utils.global_settings import static_file_folder
from ..utils.mask_create import calculate_mask, MaskProperty
from ..utils.segmentation.algorithm_base import SegmentationResult
from ..project_utils_qt.image_read_thread import ImageReaderThread
from ..project_utils_qt.main_window import BaseMainWindow
from .advanced_window import AdvancedWindow
from .batch_window import BatchWindow
from .calculation_pipeline_thread import CalculatePipelineThread
from PartSeg.tiff_image import ImageReader, Image
from PartSeg.utils.analysis.algorithm_description import analysis_algorithm_dict
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from PartSeg.utils.analysis.analysis_utils import HistoryElement, SegmentationPipelineElement, SegmentationPipeline
from .image_view import RawImageView, ResultImageView, RawImageStack, SynchronizeView
from .partseg_settings import PartSettings
from ..common_gui.custom_save_dialog import SaveDialog
from PartSeg.utils.analysis.save_functions import save_dict
from .. import CONFIG_FOLDER as CONFIG_FOLDER_BASE

CONFIG_FOLDER = os.path.join(CONFIG_FOLDER_BASE, "analysis")


class Options(QWidget):
    def __init__(self, settings: PartSettings, channel_control2: ChannelControl,
                 left_panel: RawImageView, synchronize: SynchronizeView):
        super().__init__()
        self._settings = settings
        self.left_panel = left_panel
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
        self.choose_pipe.addItems(self._settings.segmentation_pipelines.keys())
        self.choose_pipe.currentTextChanged.connect(self.choose_pipeline)
        self.choose_pipe.setToolTip("Execute chosen pipeline")
        self.save_profile_btn = QPushButton("Save segmentation profile")
        self.save_profile_btn.setToolTip("Save values from current view")
        self.save_profile_btn.clicked.connect(self.save_profile)
        self.choose_profile = QComboBox()
        self.choose_profile.addItem("<none>")
        self.choose_profile.addItems(self._settings.segmentation_profiles.keys())
        self.choose_profile.setToolTip("Select profile to restore its settings. Execute if interactive is checked")
        self.update_tooltips()
        self.choose_profile.currentTextChanged.connect(self.change_profile)
        self.interactive_use.stateChanged.connect(self.execute_btn.setDisabled)
        self.interactive_use.stateChanged.connect(self.interactive_change)
        self.algorithm_choose_widget = AlgorithmChoose(settings, analysis_algorithm_dict)
        self.algorithm_choose_widget.result.connect(self.execution_done)
        self.algorithm_choose_widget.finished.connect(self.calculation_finished)
        self.algorithm_choose_widget.value_changed.connect(self.interactive_algorithm_execute)
        self.algorithm_choose_widget.algorithm_changed.connect(self.interactive_algorithm_execute)
        # settings.mask_changed.connect(self.algorithm_choose_widget.mask_changed)
        """widgets_list = []
        for name, val in part_algorithm_dict.items():
            self.algorithm_choose.addItem(name)
            widget = InteractiveAlgorithmSettingsWidget(settings, name, val,
                                                        selector=[self.algorithm_choose, self.choose_profile])
            widgets_list.append(widget)
            widget.algorithm_thread.execution_done.connect(self.execution_done)
            widget.algorithm_thread.finished.connect(partial(self.execute_btn.setEnabled, True))
            # widget.algorithm.progress_signal.connect(self.progress_info)
            self.stack_layout.addWidget(widget)"""

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextSelectableByMouse)
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
        # self.algorithm_choose.currentIndexChanged.connect(self.stack_layout.setCurrentIndex)
        # self.algorithm_choose.currentTextChanged.connect(self.algorithm_change)
        # current_algorithm = self._settings.get("current_algorithm", self.algorithm_choose.currentText())
        # for i, el in enumerate(widgets_list):
        #     if el.name == current_algorithm:
        #        self.algorithm_choose.setCurrentIndex(i)
        #        break

    def calculation_finished(self):
        self.execute_btn.setDisabled(self.interactive_use.isChecked())
        self.interactive_use.setEnabled(True)

    def save_pipeline(self):
        history = self._settings.segmentation_history
        if not history:
            QMessageBox.information(self, "No mask created", "There is no new mask created", QMessageBox.Ok)
            return
        mask_history = []
        for el in history:
            mask = el.mask_property
            segmentation = SegmentationProfile(name="Unknown", algorithm=el.algorithm_name, values=el.algorithm_values)
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
                        self, "Already exists",
                        "Profile with this name already exist. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
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
        process_thread = CalculatePipelineThread(self._settings.image, self._settings.mask, pipeline, self)
        dial = WaitingDialog(process_thread)

        if dial.exec() and process_thread.result:
            pipeline_result = process_thread.result
            self._settings.mask = pipeline_result.mask
            self._settings.segmentation = pipeline_result.segmentation
            self._settings.full_segmentation = pipeline_result.full_segmentation
            self._settings.segmentation_history = pipeline_result.history
            self._settings.undo_segmentation_history = []
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
            """algorithm_name =  self._settings.get("current_algorithm", self.algorithm_choose.currentText())
            if algorithm_name != self.algorithm_choose.currentText():
                interactive = self.interactive_use.isChecked()
                self.interactive_use.setChecked(False)
                try:
                    self._change_profile(algorithm_name, self._settings.get(f"algorithms.{algorithm_name}"))
                except KeyError:
                    pass
                self.interactive_use.setChecked(interactive)"""
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
                        self, "Already exists",
                        "Profile with this name already exist. Overwrite?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No):
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
        self._settings.last_executed_algorithm = widget.name
        self.execute_btn.setDisabled(True)
        self.interactive_use.setDisabled(True)
        widget.execute()

    def execution_done(self, segmentation: SegmentationResult):
        self._settings.segmentation = segmentation.segmentation
        self._settings.noise_remove_image_part = segmentation.cleaned_channel
        self._settings.full_segmentation = segmentation.full_segmentation
        self.label.setText(self.sender().get_info_text())

    def showEvent(self, _event):
        self.hide_left_panel_chk.setChecked(self._settings.get_from_profile("hide_left_panel", False))


class MainMenu(QWidget):
    def __init__(self, settings: PartSettings, main_window):
        super().__init__()
        self._settings = settings
        self.open_btn = QPushButton("Open")
        self.save_btn = QPushButton("Save")
        self.advanced_btn = QPushButton("Settings and Measurement")
        self.image_adjust_btn = QPushButton("Image adjustments")
        self.mask_manager_btn = QPushButton("Mask manager")
        self.batch_processing_btn = QPushButton("Batch Processing")
        self.main_window: MainWindow = main_window

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.advanced_btn)
        layout.addWidget(self.image_adjust_btn)
        layout.addWidget(self.mask_manager_btn)
        layout.addWidget(self.batch_processing_btn)
        self.setLayout(layout)

        self.open_btn.clicked.connect(self.load_data)
        self.save_btn.clicked.connect(self.save_file)
        self.advanced_btn.clicked.connect(self.advanced_window_show)
        self.mask_manager_btn.clicked.connect(self.mask_manager)
        self.image_adjust_btn.clicked.connect(self.image_adjust_exec)
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
        base_values = self._settings.get("save_parameters", dict())
        dial = SaveDialog(save_dict, system_widget=False, base_values=base_values,
                          history=self._settings.get_path_history())
        dial.selectFile(os.path.splitext(os.path.basename(self._settings.image_path))[0])
        dial.setDirectory(self._settings.get("io.save_directory", self._settings.get("io.open_directory",
                                                                                     str(Path.home()))))
        dial.selectNameFilter(self._settings.get("io.save_filter", ""))
        if dial.exec():
            save_location, selected_filter, save_class, values = dial.get_result()
            project_info = self._settings.get_project_info()
            self._settings.set("io.save_filter", selected_filter)
            save_dir = os.path.dirname(save_location)
            self._settings.set("io.save_directory", save_dir)
            self._settings.add_path_history(save_dir)
            base_values[selected_filter] = values
            try:
                save_class.save(save_location, project_info, values)
            except ValueError as e:
                QMessageBox.warning(self, "Save error", f"Error during saving\n{e.args[0]}")

    def image_adjust_exec(self):
        dial = ImageAdjustmentDialog(self._settings.image)
        if dial.exec():
            algorithm = dial.result_val.algorithm
            dial2 = ExecuteFunctionDialog(algorithm.transform, [],
                                          {"image": self._settings.image, "arguments": dial.result_val.values}
                                          )
            if dial2.exec():
                result: Image = dial2.get_result()
                self._settings.set_project_info(ProjectTuple(result.file_path, result))
        return

    def mask_manager(self):
        if self._settings.segmentation is None:
            QMessageBox.information(self, "No segmentation", "Cannot create mask without segmentation")
            return
        dial = MaskWindow(self._settings)
        dial.exec_()

    def load_data(self):
        def exception_hook(exception):
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QMetaObject
            instance = QApplication.instance()
            if isinstance(exception, MemoryError):
                instance.warning = "Open error", f"Not enough memory to read this image: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, IOError):
                instance.warning = "Open error", f"Some problem with reading from disc: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
            elif isinstance(exception, KeyError):
                instance.warning = "Open error", f"Some problem project file: {exception}"
                QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
                print(exception, file=sys.stderr)
            else:
                raise exception

        try:
            dial = CustomLoadDialog(load_dict, history=self._settings.get_path_history())
            dial.setDirectory(self._settings.get("io.open_directory", str(Path.home())))
            dial.selectNameFilter(self._settings.get("io.open_filter", next(iter(load_dict.keys()))))
            if dial.exec_():
                result = dial.get_result()
                self._settings.set("io.open_filter", result.selected_filter)
                load_dir = os.path.dirname(result.load_location[0])
                self._settings.set("io.open_directory", load_dir)
                self._settings.add_path_history(load_dir)
                dial2 = ExecuteFunctionDialog(result.load_class.load, [result.load_location],
                                              {"metadata": {"default_spacing": self._settings.image_spacing}},
                                              exception_hook=exception_hook)
                if dial2.exec():
                    result = dial2.get_result()
                    if isinstance(result, ProjectTuple):
                        image = self._settings.verify_image(result.image, False)
                        if image:
                            if isinstance(image, Image):
                                result = result._replace(image=image)
                        else:
                            return
                self._settings.set_project_info(result)
        except ValueError as e:
            QMessageBox.warning(self, "Open error", "{}".format(e))

    def batch_window(self):
        if self.main_window.batch_window is not None:
            if self.main_window.batch_window.isVisible():
                self.main_window.batch_window.activateWindow()
            else:
                self.main_window.batch_window.show()
        else:
            self.main_window.batch_window = BatchWindow(self._settings)
            self.main_window.batch_window.show()

    def advanced_window_show(self):
        if self.main_window.advanced_window.isVisible():
            self.main_window.advanced_window.activateWindow()
        else:
            self.main_window.advanced_window.show()


class MaskWindow(QDialog):
    def __init__(self, settings: PartSettings):
        super().__init__()
        self.setWindowTitle("Mask manager")
        self.settings = settings
        main_layout = QVBoxLayout()
        self.mask_widget = MaskWidget(settings, self)
        main_layout.addWidget(self.mask_widget)
        try:
            mask_property = self.settings.get("mask_manager.mask_property")
            self.mask_widget.set_mask_property(mask_property)
        except KeyError:
            pass

        if len(settings.undo_segmentation_history) == 0:
            self.save_draw = QCheckBox("Save draw", self)
        else:
            self.save_draw = QCheckBox("Add draw", self)
        self.reset_next_btn = QPushButton("Reset Next")
        self.reset_next_btn.clicked.connect(self.reset_next_fun)
        if len(settings.undo_segmentation_history) == 0:
            self.reset_next_btn.setDisabled(True)
        self.set_next_btn = QPushButton("Set Next")
        if not self.settings.undo_segmentation_history:
            self.set_next_btn.setDisabled(True)
        self.set_next_btn.clicked.connect(self.set_next)
        self.cancel = QPushButton("Cancel", self)
        self.cancel.clicked.connect(self.close)
        self.prev_button = QPushButton(f"Previous mask ({len(settings.segmentation_history)})", self)
        if len(settings.segmentation_history) == 0:
            self.prev_button.setDisabled(True)
        self.next_button = QPushButton(f"Next mask ({len(settings.undo_segmentation_history)})", self)
        if len(settings.undo_segmentation_history) == 0:
            self.next_button.setText("Next mask (new)")
        self.next_button.clicked.connect(self.next_mask)
        self.prev_button.clicked.connect(self.prev_mask)
        op_layout = QHBoxLayout()
        op_layout.addWidget(self.save_draw)
        op_layout.addWidget(self.mask_widget.radius_information)
        main_layout.addLayout(op_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.cancel)
        button_layout.addWidget(self.set_next_btn)
        button_layout.addWidget(self.reset_next_btn)
        main_layout.addLayout(button_layout)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)
        if self.settings.undo_segmentation_history:
            mask_prop: MaskProperty = self.settings.undo_segmentation_history[-1].mask_property
            self.mask_widget.set_mask_property(mask_prop)
        self.mask_widget.values_changed.connect(self.values_changed)

    def set_next(self):
        if self.settings.undo_segmentation_history:
            self.mask_widget.set_mask_property(self.settings.undo_segmentation_history[-1].mask_property)

    def values_changed(self):
        if self.settings.undo_segmentation_history and \
                self.mask_widget.get_mask_property() == self.settings.undo_segmentation_history[-1].mask_property:
            self.next_button.setText(f"Next mask ({len(self.settings.undo_segmentation_history)})")
        else:
            self.next_button.setText("Next mask (new)")

    def reset_next_fun(self):
        self.settings.undo_segmentation_settings = []
        self.next_button.setText("Next mask (new)")
        self.reset_next_btn.setDisabled(True)

    def next_mask(self):
        algorithm_name = self.settings.last_executed_algorithm
        algorithm_values = self.settings.get(f"algorithms.{algorithm_name}")
        segmentation = self.settings.segmentation
        mask_property = self.mask_widget.get_mask_property()
        self.settings.set("mask_manager.mask_property", mask_property)
        mask = calculate_mask(mask_property, segmentation,
                              self.settings.mask, self.settings.image_spacing)
        self.settings.segmentation_history.append(
            HistoryElement.create(segmentation, self.settings.full_segmentation, self.settings.mask, algorithm_name,
                                  algorithm_values, mask_property)
        )
        if self.settings.undo_segmentation_history and \
                self.settings.undo_segmentation_history[-1].mask_property == \
                self.settings.segmentation_history[-1].mask_property:
            history = self.settings.undo_segmentation_history.pop()
            self.settings.set("current_algorithm", history.algorithm_name)
            self.settings.set(f"algorithm.{history.algorithm_name}", history.algorithm_values)
        else:
            self.settings.undo_segmentation_history = []
        self.settings.mask = mask
        self.close()

    def prev_mask(self):
        history: HistoryElement = self.settings.segmentation_history.pop()
        algorithm_name = self.settings.last_executed_algorithm
        algorithm_values = self.settings.get(f"algorithms.{algorithm_name}")
        history2 = history.replace_(algorithm_name=algorithm_name, algorithm_values=algorithm_values)
        self.settings.set("current_algorithm", history.algorithm_name)
        self.settings.set(f"algorithm.{history.algorithm_name}", history.algorithm_values)
        history.arrays.seek(0)
        seg = np.load(history.arrays)
        history.arrays.seek(0)
        self.settings.segmentation = seg["segmentation"]
        self.settings.full_segmentation = seg["full_segmentation"]
        if "mask" in seg:
            self.settings.mask = seg["mask"]
        else:
            self.settings.mask = None
        self.settings.undo_segmentation_history.append(history2)
        self.close()


class MainWindow(BaseMainWindow):
    settings_class = PartSettings
    initial_image_path = os.path.join(static_file_folder, 'initial_images', "clean_segment.tiff")

    def __init__(self, config_folder=CONFIG_FOLDER, title="PartSeg", settings=None, signal_fun=None,
                 initial_image=None):
        super().__init__(config_folder, title, settings, signal_fun)
        self.files_num = 2
        self.setMinimumWidth(600)
        self.main_menu = MainMenu(self.settings, self)
        # self.channel_control1 = ChannelControl(self.settings, name="raw_control", text="Left panel:")
        self.channel_control2 = ChannelControl(self.settings, name="result_control")
        self.raw_image = RawImageStack(self.settings,
                                       self.channel_control2)  # RawImageView(self.settings, self.channel_control1)
        self.result_image = ResultImageView(self.settings, self.channel_control2)
        self.color_bar = ColorBar(self.settings, self.raw_image.raw_image.channel_control)
        self.info_text = QLabel()
        self.raw_image.raw_image.text_info_change.connect(self.info_text.setText)
        self.result_image.text_info_change.connect(self.info_text.setText)
        self.synchronize_tool = SynchronizeView(self.raw_image.raw_image, self.result_image, self)
        # image_view_control = self.image_view.get_control_view()
        self.options_panel = Options(self.settings, self.channel_control2, self.raw_image.raw_image,
                                     self.synchronize_tool)
        # self.main_menu.image_loaded.connect(self.image_read)
        self.settings.image_changed.connect(self.image_read)
        self.advanced_window = AdvancedWindow(self.settings)
        self.batch_window = None  # BatchWindow(self.settings)

        self.multiple_files = MultipleFileWidget(self.settings, load_dict)

        if initial_image is None:
            reader = ImageReader()
            im = reader.read(self.initial_image_path)
            im.file_path = ""
            self.settings.image = im
        else:
            self.settings.image = initial_image

        icon = QIcon(os.path.join(static_file_folder, 'icons', "icon.png"))
        self.setWindowIcon(icon)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("&Open").triggered.connect(self.main_menu.load_data)
        file_menu.addAction("&Save").triggered.connect(self.main_menu.save_file)
        file_menu.addAction("Batch processing").triggered.connect(self.main_menu.batch_window)
        image_menu = menu_bar.addMenu("Image operations")
        image_menu.addAction("Image adjustment").triggered.connect(self.main_menu.image_adjust_exec)
        image_menu.addAction("Mask manager").triggered.connect(self.main_menu.mask_manager)
        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("State directory").triggered.connect(self.show_settings_directory)
        help_menu.addAction("About").triggered.connect(self.show_about_dialog)

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.main_menu, 0, 0, 1, 4)
        layout.addWidget(self.info_text, 1, 0, 1, 4, Qt.AlignHCenter)  # , 0, 4)
        layout.addWidget(self.multiple_files, 2, 0)
        layout.addWidget(self.color_bar, 2, 1)
        layout.addWidget(self.raw_image, 2, 2)  # , 0, 0)
        layout.addWidget(self.result_image, 2, 3)  # , 0, 0)
        layout.addWidget(self.options_panel, 0, 4, 3, 1)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        widget = QWidget()
        widget.setLayout(layout)
        # self.multiple_files.setHidden(True)
        self.setCentralWidget(widget)
        try:
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
        except KeyError:
            pass

    def show_settings_directory(self):
        DirectoryDialog(self.settings.json_folder_path, "Path to place where PartSeg store the data between runs").exec()

    def show_about_dialog(self):
        AboutDialog().exec()

    def image_read(self):
        self.raw_image.raw_image.set_image()
        self.raw_image.raw_image.reset_image_size()
        self.result_image.set_image()
        self.result_image.reset_image_size()
        self.options_panel.interactive_algorithm_execute()
        self.setWindowTitle(f"{self.title_base}: {os.path.basename(self.settings.image_path)}")

    def read_drop(self, paths):
        read_thread = ImageReaderThread(parent=self)

        def exception_hook(exception):
            QMessageBox.warning(self, "Read error", f"Error during image read: {exception}")

        dial = WaitingDialog(read_thread, exception_hook=exception_hook)
        if len(paths) == 1:
            file_path = paths[0]
            ext = os.path.splitext(file_path)[1]
            if ext in [".tif", ".tiff", ".lsm"]:
                read_thread.set_path(file_path)
                dial.exec()
                if read_thread.image:
                    self.settings.image = read_thread.image
            elif ext in [".tgz", ".tbz2", ".gz", ".bz2"]:
                self.settings.load_project(file_path)
        elif len(paths) == 2:
            name1, name2 = [os.path.basename(os.path.splitext(x)[0]) for x in paths]
            if name1.endswith("_mask"):
                read_thread.set_path(paths[1], paths[0])
            elif name2.endswith("_mask"):
                read_thread.set_path(paths[0], paths[1])
            else:
                return
            dial.exec()
            if read_thread.image:
                self.settings.image = read_thread.image

    def event(self, event: QEvent):
        if event.type() == QEvent.WindowActivate:
            self.multiple_files.setVisible(self.settings.get("multiple_files", False))
        return super().event(event)

    def closeEvent(self, event):
        # print(self.settings.dump_view_profiles())
        # print(self.settings.segmentation_dict["default"].my_dict)
        self.settings.set_in_profile("main_window_geometry", bytes(self.saveGeometry().toHex()).decode('ascii'))
        self.options_panel.algorithm_choose_widget.recursive_get_values()
        if self.batch_window is not None:
            if self.batch_window.is_working():
                ret = QMessageBox.warning(self, "Batch work", "Batch work is not finished. "
                                                              "Would you like to terminate it?",
                                          QMessageBox.No | QMessageBox.Yes)
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
