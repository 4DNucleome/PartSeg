import json
import logging
import os
import re
from functools import partial
from io import BytesIO

import tifffile as tif
import numpy as np
import SimpleITK as sitk
import appdirs
from PyQt5.QtCore import Qt, QByteArray, QEvent
from PyQt5.QtGui import QIcon, QKeyEvent
from PyQt5.QtWidgets import QLabel, QWidget, QPushButton, QHBoxLayout, QVBoxLayout, QGridLayout, \
    QFileDialog, QMessageBox, QCheckBox, QComboBox, QStackedLayout, QInputDialog, QDialog, QSpinBox, QAbstractSpinBox

from common_gui.channel_control import ChannelControl
from common_gui.mask_widget import MaskWidget
from common_gui.stack_image_view import ColorBar
from common_gui.waiting_dialog import WaitingDialog
from partseg2.advanced_window import AdvancedWindow
from partseg2.batch_window import BatchWindow
from partseg2.calculation_pipeline_thread import CalculatePipelineThread
from partseg2.interpolate_dialog import InterpolateDialog
from partseg2.interpolate_thread import InterpolateThread
from project_utils.algorithms_description import InteractiveAlgorithmSettingsWidget
from project_utils.error_dialog import ErrorDialog
from project_utils.global_settings import static_file_folder
from project_utils.image_operations import dilate, erode, RadiusType
from project_utils.image_read_thread import ImageReaderThread
from project_utils.main_window import BaseMainWindow
from project_utils.mask_create import calculate_mask, MaskProperty
from .partseg_settings import PartSettings, load_project, save_project, save_labeled_image
from .partseg_utils import HistoryElement, SegmentationPipelineElement, SegmentationPipeline
from .image_view import RawImageView, ResultImageView, RawImageStack, SynchronizeView
from .algorithm_description import part_algorithm_dict, SegmentationProfile
from tiff_image import ImageReader

app_name = "PartSeg2"
app_lab = "LFSG"
config_folder = appdirs.user_data_dir(app_name, app_lab)


class Options(QWidget):
    def __init__(self, settings: PartSettings, channel_control2: ChannelControl,
                 left_panel: RawImageView, synchronize: SynchronizeView):
        super().__init__()
        self._settings = settings
        self.left_panel = left_panel
        self._ch_control2 = channel_control2
        self.hide_left_panel_chk = QCheckBox("Hide left panel")
        self.hide_left_panel_chk.stateChanged.connect(self.hide_left_panel)
        self.synchronize_checkbox = QCheckBox("Synchronize view")
        self.synchronize_checkbox.stateChanged.connect(synchronize.set_synchronize)
        self.stack_layout = QStackedLayout()
        self.algorithm_choose = QComboBox()
        self.interactive_use = QCheckBox("Interactive use")
        self.execute_btn = QPushButton("Execute")
        self.execute_btn.clicked.connect(self.execute_algorithm)
        self.save_pipe_btn = QPushButton("Save pipeline")
        self.save_pipe_btn.clicked.connect(self.save_pipeline)
        self.choose_pipe = QComboBox()
        self.choose_pipe.addItem("<none>")
        self.choose_pipe.addItems(self._settings.segmentation_pipelines.keys())
        self.choose_pipe.currentTextChanged.connect(self.choose_pipeline)
        self.save_profile_btn = QPushButton("Save segmentation profile")
        self.choose_profile = QComboBox()
        self.choose_profile.addItem("<none>")
        self.choose_profile.addItems(self._settings.segmentation_profiles.keys())
        self.choose_profile.setToolTip("Select profile to restore its settings")
        self.update_tooltips()
        self.choose_profile.currentTextChanged.connect(self.change_profile)
        self.interactive_use.stateChanged.connect(self.execute_btn.setDisabled)
        self.interactive_use.stateChanged.connect(self.interactive_change)
        self.save_profile_btn.clicked.connect(self.save_profile)
        widgets_list = []
        for name, val in part_algorithm_dict.items():
            self.algorithm_choose.addItem(name)
            widget = InteractiveAlgorithmSettingsWidget(settings, name, val,
                                                        selector=[self.algorithm_choose, self.choose_profile])
            widgets_list.append(widget)
            widget.algorithm_thread.execution_done[np.ndarray, np.ndarray].connect(self.execution_done)
            widget.algorithm_thread.finished.connect(partial(self.execute_btn.setEnabled, True))
            # widget.algorithm.progress_signal.connect(self.progress_info)
            self.stack_layout.addWidget(widget)

        self.label = QLabel()
        self.label.setWordWrap(True)
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
        layout.addWidget(self.algorithm_choose)
        layout.addLayout(self.stack_layout)
        layout.addWidget(self.label)
        layout.addStretch(1)
        layout2.addWidget(self.hide_left_panel_chk)
        layout2.addWidget(self.synchronize_checkbox)
        layout.addLayout(layout2)
        layout.addWidget(self._ch_control2)
        layout.setSpacing(0)
        self.setLayout(layout)
        self.algorithm_choose.currentIndexChanged.connect(self.stack_layout.setCurrentIndex)
        self.algorithm_choose.currentTextChanged.connect(self.algorithm_change)
        current_algorithm = self._settings.get("current_algorithm", self.algorithm_choose.currentText())
        for i, el in enumerate(widgets_list):
            if el.name == current_algorithm:
                self.algorithm_choose.setCurrentIndex(i)
                break

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
        widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.currentWidget()
        current_segmentation = SegmentationProfile(name="Unknown", algorithm=widget.name, values=widget.get_values())

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
        if text =="<none>":
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
            self._change_profile(pipeline.segmentation.algorithm, pipeline.segmentation.values)
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
            algorithm_name =  self._settings.get("current_algorithm", self.algorithm_choose.currentText())
            if algorithm_name != self.algorithm_choose.currentText():
                interactive = self.interactive_use.isChecked()
                self.interactive_use.setChecked(False)
                try:
                    self._change_profile(algorithm_name, self._settings.get(f"algorithms.{algorithm_name}"))
                except KeyError:
                    pass
                self.interactive_use.setChecked(interactive)
        return super().event(event)

    def keyPressEvent(self, event: QKeyEvent):
        if (event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return) and (event.modifiers() == Qt.ControlModifier):
            self.execute_btn.click()

    def save_profile(self):
        widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.currentWidget()
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
            self._settings.set(f"segmentation_profiles.{text}", resp)
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
        profile = self._settings.get(f"segmentation_profiles.{val}")
        self._change_profile(profile.algorithm, profile.values)
        self.choose_profile.blockSignals(True)
        self.choose_profile.setCurrentIndex(0)
        self.choose_profile.blockSignals(False)
        self.interactive_use.setChecked(interactive)

    def _change_profile(self, name, values):
        for i in range(self.stack_layout.count()):
            widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.widget(i)
            if widget.name == name:
                self.algorithm_choose.setCurrentIndex(i)
                widget.set_values(values)
                break
    @property
    def segmentation(self):
        return self._settings.segmentation

    @segmentation.setter
    def segmentation(self, val):
        self._settings.segmentation = val

    def image_changed(self):
        self.segmentation = None

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

    def image_changed_exec(self):
        if self.interactive:
            self.execute_algorithm()

    def execute_algorithm(self):
        widget: InteractiveAlgorithmSettingsWidget = self.stack_layout.currentWidget()
        self._settings.set("last_executed_algorithm", widget.name)
        self.execute_btn.setDisabled(True)
        widget.execute()

    def execution_done(self, segmentation, full_segmentation):
        self.segmentation = segmentation
        self._settings.full_segmentation = full_segmentation
        self.label.setText(self.sender().get_info_text())

    def showEvent(self, _event):
        self.hide_left_panel_chk.setChecked(self._settings.get_from_profile("hide_left_panel", False))


class MainMenu(QWidget):
    def __init__(self, settings: PartSettings, main_window):
        super().__init__()
        self._settings = settings
        self.open_btn = QPushButton("Open")
        self.save_btn = QPushButton("Save")
        self.advanced_btn = QPushButton("Advanced")
        self.interpolate_btn = QPushButton("Interpolate")
        self.mask_manager_btn = QPushButton("Mask Manager")
        self.batch_processing_btn = QPushButton("Batch Processing")
        self.main_window: MainWindow = main_window

        layout = QHBoxLayout()
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.open_btn)
        layout.addWidget(self.save_btn)
        layout.addWidget(self.advanced_btn)
        layout.addWidget(self.interpolate_btn)
        layout.addWidget(self.mask_manager_btn)
        layout.addWidget(self.batch_processing_btn)
        self.setLayout(layout)

        self.open_btn.clicked.connect(self.load_data)
        self.save_btn.clicked.connect(self.save_file)
        self.advanced_btn.clicked.connect(self.advanced_window_show)
        self.mask_manager_btn.clicked.connect(self.mask_manager)
        self.interpolate_btn.clicked.connect(self.interpolate_exec)
        self.batch_processing_btn.clicked.connect(self.batch_window)

    def interpolate_exec(self):
        dialog = InterpolateDialog(self._settings.image_spacing)
        if dialog.exec():
            scale_factor = dialog.get_zoom_factor()
            print(scale_factor)
            interp_ob = InterpolateThread()
            dial = WaitingDialog(interp_ob)
            args = [self._settings.image]
            if self._settings.mask is not None:
                mask = self._settings.mask.astype(np.uint8)
                mask[mask > 0] = 255
                args.append(mask)
            interp_ob.set_arrays(args)
            interp_ob.set_scaling(scale_factor)
            if dial.exec():
                self._settings.image = interp_ob.result[0], self._settings.image_path
                if len(interp_ob.result) == 2:
                    self._settings.mask = interp_ob.result[1] > 128
                self._settings.image_spacing = [x/y for x,y in zip(self._settings.image_spacing, scale_factor)]
            else:
                if interp_ob.isRunning():
                    interp_ob.terminate()
            #self.settings.rescale_image(dialog.get_zoom_factor())

    def mask_manager(self):
        if self._settings.segmentation is None:
            QMessageBox.information(self, "No segmentation", "Cannot create mask without segmentation")
            return
        dial = MaskWindow(self._settings)
        dial.exec_()

    def load_data(self):
        try:
            dial = QFileDialog(self, "Load data")
            dial.setDirectory(self._settings.get("io.open_directory", ""))
            dial.setFileMode(QFileDialog.ExistingFile)
            filters = ["raw image (*.tiff *.tif *.lsm)", "image with mask (*.tiff *.tif *.lsm)",
                       "mask to image (*.tiff *.tif *.lsm)",
                       "saved project (*.tgz *.tbz2 *.gz *.bz2)", "Profiles (*.json)"]
            # dial.setFilters(filters)
            dial.setNameFilters(filters)
            dial.selectNameFilter(self._settings.get("io.open_filter", filters[0]))
            if dial.exec_():
                file_path = str(dial.selectedFiles()[0])
                self._settings.set("io.open_directory", os.path.dirname(str(file_path)))
                selected_filter = str(dial.selectedNameFilter())
                self._settings.set("io.open_filter", selected_filter)
                logging.debug("open file: {}, filter {}".format(file_path, selected_filter))
                # TODO maybe something better. Now main window have to be parent
                read_thread = ImageReaderThread(parent=self)
                dial = WaitingDialog(read_thread)
                if selected_filter == "raw image (*.tiff *.tif *.lsm)":
                    read_thread.set_path(file_path)
                    dial.exec()
                    self._settings.image = read_thread.image
                    #self._settings.image_spacing = list(np.array([70, 70 ,210]) * 0.1**9)
                elif selected_filter == "mask to image (*.tiff *.tif *.lsm)":
                    im = tif.imread(file_path)
                    self._settings.mask = im
                elif selected_filter == "image with mask (*.tiff *.tif *.lsm)":
                    extension = os.path.splitext(file_path)
                    if extension == ".json":
                        with open(file_path) as ff:
                            info_dict = json.load(ff)
                        read_thread.set_path(info_dict["image"], info_dict["mask"])
                        dial.exec()
                        self._settings.image = read_thread.image
                    else:
                        org_name = os.path.basename(file_path)
                        mask_dial = QFileDialog(self, "Load mask for {}".format(org_name))
                        filters = ["mask (*.tiff *.tif *.lsm)"]
                        mask_dial.setNameFilters(filters)
                        if mask_dial.exec_():
                            read_thread.set_path(file_path, mask_dial.selectedFiles()[0])
                            dial.exec()
                            self._settings.image = read_thread.image
                elif selected_filter == "saved project (*.tgz *.tbz2 *.gz *.bz2)":
                    load_project(file_path, self._settings)
                    # self.segment.threshold_updated()
                elif selected_filter == "Profiles (*.json)":
                    self._settings.load_profiles(file_path)
                else:
                    # noinspection PyCallByClass
                    _ = QMessageBox.warning(self, "Load error", "Function do not implemented yet")
                    return
        except (IOError, MemoryError) as e:
            QMessageBox.warning(self, "Open error", "Exception occurred {}".format(e))
        except Exception as e:
            ErrorDialog(e, "Image read").exec()


    def batch_window(self):
        if self.main_window.batch_window.isVisible():
            self.main_window.batch_window.activateWindow()
        else:
            self.main_window.batch_window.show()

    def save_file(self):
        try:
            dial = QFileDialog(self, "Save data")
            dial.setDirectory(self._settings.get("io.save_directory", self._settings.get("io.open_directory", "")))
            dial.setFileMode(QFileDialog.AnyFile)
            filters = ["Project (*.tgz *.tbz2 *.gz *.bz2)", "Labeled image (*.tif)", "Mask in tiff (*.tif)",
                       "Mask for itk-snap (*.img)", "Data for chimera (*.cmap)", "Image (*.tiff)", "Profiles (*.json)",
                       "Segmented data in xyz (*.xyz)"]
            dial.setAcceptMode(QFileDialog.AcceptSave)
            dial.setNameFilters(filters)
            default_name = os.path.splitext(os.path.basename(self._settings.image.file_path))[0]
            dial.selectFile(default_name)
            dial.selectNameFilter(self._settings.get("io.save_filter", ""))
            if dial.exec_():
                file_path = str(dial.selectedFiles()[0])
                selected_filter = str(dial.selectedNameFilter())
                self._settings.set("io.save_filter", selected_filter)
                self._settings.set("io.save_directory", os.path.dirname(file_path))
                if os.path.splitext(file_path)[1] == '':
                    ext = re.search(r'\(\*(\.\w+)', selected_filter).group(1)
                    file_path += ext
                    if os.path.exists(file_path):
                        # noinspection PyCallByClass
                        ret = QMessageBox.warning(self, "File exist", os.path.basename(file_path) +
                                                  " already exists.\nDo you want to replace it?",
                                                  QMessageBox.No | QMessageBox.Yes)
                        if ret == QMessageBox.No:
                            self.save_file()
                            return

                if selected_filter == "Project (*.tgz *.tbz2 *.gz *.bz2)":
                    if self._settings.segmentation is None:
                        QMessageBox.warning(self, "No segmentation", "Cannot save project with no segmentation",
                                            QMessageBox.Ok)
                        return
                    self._settings.save_project(file_path)
                elif selected_filter == "Labeled image (*.tif)":
                    save_labeled_image(file_path, self._settings)

                elif selected_filter == "Mask in tiff (*.tif)":
                    segmentation = self._settings.segmentation
                    segmentation = np.array(segmentation > 0).astype(np.uint8)
                    tif.imsave(file_path, segmentation)
                elif selected_filter == "Mask for itk-snap (*.img)":
                    segmentation = sitk.GetImageFromArray(self.segment.get_segmentation())
                    sitk.WriteImage(segmentation, file_path)
                elif selected_filter == "Data for chimera (*.cmap)":
                    if not np.any(self.segment.get_segmentation()):
                        QMessageBox.warning(self, "No object", "There is no component to export to cmap")
                        return
                    ob = CmapSave(file_path, self.settings, self.segment)
                    ob.exec_()
                elif selected_filter == "Image (*.tiff)":
                    image = self.settings.image
                    tif.imsave(file_path, image)
                elif selected_filter == "Profiles (*.json)":
                    self.settings.dump_profiles(file_path)
                elif selected_filter == "Segmented data in xyz (*.xyz)":
                    save_to_xyz(file_path, self.settings, self.segment)
                else:
                    # noinspection PyCallByClass
                    _ = QMessageBox.critical(self, "Save error", "Option unknown")
        except IOError as e:
            QMessageBox.warning(self, "Open error", "Exception occurred {}".format(e))

    def advanced_window_show(self):
        if self.main_window.advanced_window.isVisible():
            self.main_window.advanced_window.activateWindow()
        else:
            self.main_window.advanced_window.show()


class MaskWindow(QDialog):
    def __init__(self, settings:PartSettings):
        super(MaskWindow, self).__init__()
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
        algorithm_name = self.settings.get("last_executed_algorithm")
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
                self.settings.undo_segmentation_history[-1] == self.settings.segmentation_history[-1]:
            self.settings.undo_segmentation_history.pop()
        else:
            self.settings.undo_segmentation_history = []
        self.settings.mask = mask
        self.close()

    def prev_mask(self):
        history: HistoryElement = self.settings.segmentation_history.pop()
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
        self.settings.undo_segmentation_history.append(history)
        self.close()


class MainWindow(BaseMainWindow):
    def __init__(self, title, signal_fun=None):
        super().__init__(signal_fun)
        self.setWindowTitle(title)
        self.title = title
        self.setMinimumWidth(600)
        self.settings = PartSettings(os.path.join(config_folder, "settings.json"))
        if os.path.exists(os.path.join(config_folder, "settings.json")):
            self.settings.load()
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
        self.batch_window = BatchWindow(self.settings)

        reader =ImageReader()
        im = reader.read(os.path.join(static_file_folder, 'initial_images', "clean_segment.tiff"))
        self.settings.image = im

        icon = QIcon(os.path.join(static_file_folder, 'icons', "icon.png"))
        self.setWindowIcon(icon)

        layout = QGridLayout()
        layout.setSpacing(0)
        layout.addWidget(self.main_menu, 0, 0, 1, 3)
        layout.addWidget(self.info_text, 1, 0, 1, 3, Qt.AlignHCenter)  # , 0, 4)
        layout.addWidget(self.color_bar, 2, 0)
        layout.addWidget(self.raw_image, 2, 1)  # , 0, 0)
        layout.addWidget(self.result_image, 2, 2)  # , 0, 0)
        layout.addWidget(self.options_panel, 0, 3, 3, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
        try:
            geometry = self.settings.get_from_profile("main_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
        except KeyError:
            pass

    def image_read(self):
        self.raw_image.raw_image.set_image()
        self.raw_image.raw_image.reset_image_size()
        self.result_image.set_image()
        self.result_image.reset_image_size()
        self.options_panel.image_changed_exec()
        self.setWindowTitle(f"PartSeg: {self.settings.image_path}")

    def read_drop(self, paths):
        assert len(paths) == 1
        ext = os.path.splitext(paths[0])[1]
        read_thread = ImageReaderThread(parent=self)
        if ext in [".tif", ".tiff", ".lsm"]:
            read_thread.set_path(paths[0])
            dial = WaitingDialog(read_thread)
            dial.exec()
            if read_thread.image:
                self.settings.image = read_thread.image

    def closeEvent(self, event):
        # print(self.settings.dump_view_profiles())
        # print(self.settings.segmentation_dict["default"].my_dict)
        self.settings.set_in_profile("main_window_geometry", bytes(self.saveGeometry().toHex()).decode('ascii'))
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
