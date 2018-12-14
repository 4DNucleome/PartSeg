import logging
from copy import copy, deepcopy

from common_gui.universal_gui_part import right_label

from partseg_old.batch_processing.calculation_plan import CalculationPlan, MaskCreate, MaskUse, Operations, CmapProfile, \
    MaskSuffix, MaskSub, \
    MaskFile, ProjectSave, PlanChanges, NodeType, ChooseChanel, MaskIntersection, MaskSum, MaskSave, ImageSave, \
    XYZSave
from partseg_old.io_functions import GaussUse
from partseg_old.profile_export import ExportDialog, ImportDialog
from partseg_old.segment import SegmentationProfile
from partseg_old.statistics_calculation import StatisticProfile
from partseg_old.qt_import import QWidget, QSplitter, QTreeWidget, QTreeWidgetItem, pyqtSignal, QPushButton, QCheckBox, \
    QListWidget, QLineEdit, QSpinBox, QTextEdit, QGroupBox, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, \
    QInputDialog, QMessageBox, QFileDialog, QDialog, QComboBox, str_type, QCompleter, Qt

group_sheet = "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} " \
              "QGroupBox::title {subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px;}"


class TwoMaskDialog(QDialog):
    def __init__(self, mask_names):
        """
        :type mask_names: set
        :param mask_names: iterable collection of all available mask names
        """
        super(TwoMaskDialog, self).__init__()
        self.mask_names = mask_names
        completer = QCompleter(list(mask_names))
        completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.setWindowTitle("Masks name choose")
        self.mask1_name = QLineEdit()
        self.mask2_name = QLineEdit()
        self.cancel_btn = QPushButton("Cancel")
        self.ok_btn = QPushButton("Ok")

        self.mask1_name.setCompleter(completer)
        self.mask1_name.textChanged.connect(self.text_changed)
        self.mask2_name.setCompleter(completer)
        self.mask2_name.textChanged.connect(self.text_changed)
        self.cancel_btn.clicked.connect(self.close)
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setDisabled(True)

        layout = QGridLayout()
        layout.addWidget(right_label("Mask 1 name:"), 0, 0)
        layout.addWidget(self.mask1_name, 0, 1)
        layout.addWidget(right_label("Mask 2 name:"), 1, 0)
        layout.addWidget(self.mask2_name, 1, 1)
        layout.addWidget(self.cancel_btn, 2, 0)
        layout.addWidget(self.ok_btn, 2, 1)
        self.setLayout(layout)

    def text_changed(self):
        text1, text2 = self.get_result()
        if text1 == "" or text2 == "" or text1 not in self.mask_names or text2 not in self.mask_names:
            self.ok_btn.setDisabled(True)
            return
        else:
            self.ok_btn.setDisabled(text1 == text2)

    def get_result(self):
        text1 = str(self.mask1_name.text()).strip()
        text2 = str(self.mask2_name.text()).strip()
        return text1, text2


class CreatePlan(QWidget):
    """
    :type settings: Settings
    """

    plan_created = pyqtSignal()
    plan_node_changed = pyqtSignal()

    def __init__(self, settings):
        super(CreatePlan, self).__init__()
        self.settings = settings
        self.plan = PlanPreview(self)
        self.save_plan_btn = QPushButton("Save plan")
        self.clean_plan_btn = QPushButton("Clean plan")
        self.remove_btn = QPushButton("Remove")
        # self.forgot_mask_btn = QPushButton("Forgot mask")
        # self.cmap_save_btn = QPushButton("Save to cmap")
        self.choose_channel_btn = QPushButton("Choose channel")
        self.update_element_btn = QCheckBox("Update element")
        self.save_choose = QComboBox()
        self.save_choose.addItems(["<none>", "Mask", "Image", "Project", "xyz file", "Chimera cmap"])
        self.director_save_chk = QCheckBox("Save in directory")
        self.director_save_chk.setToolTip("Create directory using file name an put result file inside this directory")
        self.project_save_btn = QPushButton("Save")
        # self.save_mask_btn = QPushButton("Save mask")
        # self.forgot_mask_btn.setToolTip("Return to state on begin")
        self.segment_profile = QListWidget()
        self.generate_mask = QPushButton("Generate mask")
        self.generate_mask.setToolTip("Mask need to have unique name")
        self.mask_name = QLineEdit()
        self.base_mask_name = QLineEdit()
        self.swap_mask_name = QLineEdit()
        self.mapping_file_button = QPushButton("Mask mapping file")
        self.swap_mask_name_button = QPushButton("Name Substitution")
        self.suffix_mask_name_button = QPushButton("Name suffix")
        self.reuse_mask = QPushButton("Reuchoose_channelse mask")
        self.set_mask_name = QPushButton("Set mask name")
        self.intersect_mask_btn = QPushButton("Mask intersection")
        self.sum_mask_btn = QPushButton("Mask sum")
        self.chanel_pos = QSpinBox()
        self.chanel_pos.setRange(0, 100)
        self.chanel_num = QSpinBox()
        self.chanel_num.setRange(0, 10)
        self.expected_node_type = None
        self.save_constructor = None
        self.channels_used = False

        self.project_segmentation = QPushButton("Segmentation\nfrom project")
        self.project_segmentation.clicked.connect(self.segmentation_from_project)

        self.chose_profile = QPushButton("Segment Profile")
        self.get_big = QPushButton("Leave the biggest")
        self.statistic_list = QListWidget(self)
        self.statistic_name_prefix = QLineEdit(self)
        self.add_calculation = QPushButton("Add statistic calculation")
        self.information = QTextEdit()
        self.information.setReadOnly(True)

        self.protect = False
        self.mask_set = set()
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.dilate_radius_spin = QSpinBox()
        self.dilate_radius_spin.setRange(-50, 50)

        self.save_choose.currentIndexChanged[str_type].connect(self.save_changed)
        self.statistic_list.currentTextChanged.connect(self.show_statistics)
        self.segment_profile.currentTextChanged.connect(self.show_segment)
        self.statistic_list.currentTextChanged[str_type].connect(self.show_statistics_info)
        self.segment_profile.currentTextChanged[str_type].connect(self.show_segment_info)
        self.mask_name.textChanged[str_type].connect(self.mask_name_changed)
        self.generate_mask.clicked.connect(self.create_mask)
        self.reuse_mask.clicked.connect(self.use_mask)
        self.clean_plan_btn.clicked.connect(self.clean_plan)
        self.remove_btn.clicked.connect(self.remove_element)
        self.base_mask_name.textChanged.connect(self.mask_text_changed)
        self.swap_mask_name.textChanged.connect(self.mask_text_changed)
        self.mask_name.textChanged.connect(self.mask_text_changed)
        self.chose_profile.clicked.connect(self.add_segmentation)
        self.get_big.clicked.connect(self.add_leave_biggest)
        self.add_calculation.clicked.connect(self.add_statistics)
        self.save_plan_btn.clicked.connect(self.add_calculation_plan)
        # self.forgot_mask_btn.clicked.connect(self.forgot_mask)
        # self.cmap_save_btn.clicked.connect(self.save_to_cmap)
        self.swap_mask_name_button.clicked.connect(self.mask_by_substitution)
        self.suffix_mask_name_button.clicked.connect(self.mask_by_suffix)
        self.mapping_file_button.clicked.connect(self.mask_by_mapping)
        self.project_save_btn.clicked.connect(self.save_to_project)
        self.update_element_btn.stateChanged.connect(self.mask_text_changed)
        self.update_element_btn.stateChanged.connect(self.show_statistics)
        self.update_element_btn.stateChanged.connect(self.show_segment)
        self.update_element_btn.stateChanged.connect(self.update_names)
        self.choose_channel_btn.clicked.connect(self.choose_channel)
        self.intersect_mask_btn.clicked.connect(self.mask_intersect)
        self.sum_mask_btn.clicked.connect(self.mask_sum)
        plan_box = QGroupBox("Calculate plan:")
        lay = QVBoxLayout()
        lay.addWidget(self.plan)
        bt_lay = QGridLayout()
        bt_lay.addWidget(self.save_plan_btn, 0, 0)
        bt_lay.addWidget(self.clean_plan_btn, 0, 1)
        bt_lay.addWidget(self.remove_btn, 1, 0)
        bt_lay.addWidget(self.update_element_btn, 1, 1)
        lay.addLayout(bt_lay)
        plan_box.setLayout(lay)
        plan_box.setStyleSheet(group_sheet)

        other_box = QGroupBox("Other operations:")
        bt_lay = QGridLayout()
        bt_lay.addWidget(right_label("Chanel pos:"), 0, 0)
        bt_lay.addWidget(self.chanel_pos, 0, 1)
        bt_lay.addWidget(right_label("Chanel num:"), 1, 0)
        bt_lay.addWidget(self.chanel_num, 1, 1)
        bt_lay.addWidget(self.choose_channel_btn, 4, 0, 1, 2)
        # bt_lay.addWidget(self.forgot_mask_btn, 1, 0)
        bt_lay.addWidget(self.save_choose, 5, 0, 1, 2)
        bt_lay.addWidget(self.director_save_chk, 6, 0, 1, 2)
        bt_lay.addWidget(self.project_save_btn, 7, 0, 1, 2)
        bt_lay.addWidget(self.project_segmentation, 8, 0, 1, 2)
        other_box.setLayout(bt_lay)
        other_box.setStyleSheet(group_sheet)

        file_mask_box = QGroupBox("Mask from file")
        file_mask_box.setStyleSheet(group_sheet)
        lay = QGridLayout()
        # lay.addWidget(QLabel("Mask name:"), 0, 0)
        # lay.addWidget(self.file_mask_name, 0, 1, 1, 2)
        lay.addWidget(self.mapping_file_button, 2, 0, 1, 2)
        lay.addWidget(QLabel("Suffix/Sub string:"), 3, 0)
        lay.addWidget(QLabel("Replace:"), 3, 1)
        lay.addWidget(self.base_mask_name, 4, 0)
        lay.addWidget(self.swap_mask_name, 4, 1)
        lay.addWidget(self.suffix_mask_name_button, 5, 0)
        lay.addWidget(self.swap_mask_name_button, 5, 1)
        file_mask_box.setLayout(lay)

        segmentation_mask_box = QGroupBox("Mask from segmentation")
        segmentation_mask_box.setStyleSheet(group_sheet)
        lay = QGridLayout()
        # lay.addWidget(right_label("Mask name:"), 0, 0)
        # lay.addWidget(self.mask_name, 0, 1, 1, 2)
        lay.addWidget(right_label("Dilate radius"), 1, 0)
        lay.addWidget(self.dilate_radius_spin, 1, 1)
        lay.addWidget(self.generate_mask, 1, 2)
        segmentation_mask_box.setLayout(lay)

        mask_box = QGroupBox("Mask:")
        mask_box.setStyleSheet(group_sheet)
        lay = QGridLayout()
        name_lay = QHBoxLayout()
        name_lay.addWidget(right_label("Mask name:"))
        name_lay.addWidget(self.mask_name)
        lay.addLayout(name_lay, 0, 0, 1, 2)
        lay.addWidget(self.set_mask_name, 1, 0)
        lay.addWidget(self.reuse_mask, 1, 1)
        lay.addWidget(self.intersect_mask_btn, 2, 0)
        lay.addWidget(self.sum_mask_btn, 2, 1)
        lay.addWidget(file_mask_box, 3, 0, 1, 2)
        lay.addWidget(segmentation_mask_box, 4, 0, 1, 2)
        mask_box.setLayout(lay)

        segment_box = QGroupBox("Segmentation")
        segment_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.segment_profile)
        lay.addWidget(self.chose_profile)
        lay.addWidget(self.get_big)
        segment_box.setLayout(lay)

        statistic_box = QGroupBox("Statistics")
        statistic_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.statistic_list)
        lab = QLabel("Name prefix:")
        lab.setToolTip("Prefix added before each column name")
        line_lay = QHBoxLayout()
        line_lay.addWidget(lab)
        line_lay.addWidget(self.statistic_name_prefix)
        lay.addLayout(line_lay)
        lay.addWidget(self.add_calculation)
        statistic_box.setLayout(lay)

        info_box = QGroupBox("Information")
        info_box.setStyleSheet(group_sheet)
        lay = QVBoxLayout()
        lay.addWidget(self.information)
        info_box.setLayout(lay)

        layout = QGridLayout()
        fst_col = QVBoxLayout()
        fst_col.addWidget(plan_box)
        fst_col.addWidget(mask_box)
        layout.addLayout(fst_col, 0, 0, 0, 1)
        # layout.addWidget(plan_box, 0, 0, 3, 1)
        # layout.addWidget(mask_box, 3, 0, 2, 1)
        # layout.addWidget(segmentation_mask_box, 1, 1)
        layout.addWidget(segment_box, 0, 2)
        layout.addWidget(other_box, 0, 1)
        layout.addWidget(statistic_box, 1, 1, 1, 2)
        layout.addWidget(info_box, 3, 1, 1, 2)
        self.setLayout(layout)

        self.reuse_mask.setDisabled(True)
        self.generate_mask.setDisabled(True)
        self.chose_profile.setDisabled(True)
        self.add_calculation.setDisabled(True)
        self.swap_mask_name_button.setDisabled(True)
        self.suffix_mask_name_button.setDisabled(True)
        self.mapping_file_button.setDisabled(True)

        self.mask_allow = False
        self.segment_allow = False
        self.file_mask_allow = False
        self.node_type = NodeType.root
        self.node_name = ""
        self.plan_node_changed.connect(self.mask_text_changed)
        self.plan.changed_node.connect(self.node_type_changed)
        self.plan_node_changed.connect(self.show_segment)
        self.plan_node_changed.connect(self.show_statistics)
        self.node_type_changed()

    def save_changed(self, text):
        text = str(text)
        if text == "<none>":
            self.project_save_btn.setText("Save")
            self.project_save_btn.setToolTip("Choose file type")
            self.expected_node_type = None
            self.save_constructor = None
        elif text == "Mask":
            self.project_save_btn.setText("Save to Mask")
            self.project_save_btn.setToolTip("Choose mask create in plan view")
            self.expected_node_type = NodeType.mask
            self.save_constructor = MaskSave
        elif text == "Image":
            self.project_save_btn.setText("Save image")
            self.project_save_btn.setToolTip("Choose root or channel choose in plan view")
            self.expected_node_type = NodeType.root
            self.save_constructor = ImageSave
        elif text == "Project":
            self.project_save_btn.setText("Save to project")
            self.project_save_btn.setToolTip("Choose segmentation in plan view")
            self.expected_node_type = NodeType.segment
            self.save_constructor = ProjectSave
        elif text == "xyz file":
            self.project_save_btn.setText("Save to xyz")
            self.project_save_btn.setToolTip("Choose segmentation in plan view")
            self.expected_node_type = NodeType.segment
            self.save_constructor = XYZSave
        elif text == "Chimera cmap":
            self.project_save_btn.setText("Save to cmap")
            self.project_save_btn.setToolTip("Choose segmentation in plan view")
            self.expected_node_type = NodeType.segment
            self.save_constructor = CmapProfile
        else:
            self.project_save_btn.setText("Save")
            self.project_save_btn.setToolTip("Choose file type")
            self.expected_node_type = None
            self.save_constructor = None
            logging.error("[CreatePlan.save_changed] unknown option {}".format(text))
        self.save_activate()

    def save_activate(self):
        self.project_save_btn.setDisabled(True)
        if self.node_type == self.expected_node_type:
            self.project_save_btn.setEnabled(True)
            return
        if self.node_type == NodeType.channel_choose and self.expected_node_type == NodeType.root:
            self.project_save_btn.setEnabled(True)

    def segmentation_from_project(self):
        self.calculation_plan.add_step(Operations.segment_from_project)
        self.plan.update_view()

    def update_names(self):
        if self.update_element_btn.isChecked():
            self.chose_profile.setText("Replace Segment Profile")
            self.add_calculation.setText("Replace statistic calculation")
            self.generate_mask.setText("Replace mask")
        else:
            self.chose_profile.setText("Segment Profile")
            self.add_calculation.setText("Add statistic calculation")
            self.generate_mask.setText("Generate mask")

    def node_type_changed(self):
        # self.cmap_save_btn.setDisabled(True)
        self.project_save_btn.setDisabled(True)
        self.project_segmentation.setDisabled(True)
        self.choose_channel_btn.setDisabled(True)
        self.node_name = ""
        if self.plan.currentItem() is None:
            self.mask_allow = False
            self.file_mask_allow = False
            self.segment_allow = False
            self.remove_btn.setDisabled(True)
            self.plan_node_changed.emit()
            logging.debug("[node_type_changed] return")
            return
        node_type = self.calculation_plan.get_node_type()
        self.node_type = node_type
        if node_type in [NodeType.file_mask, NodeType.mask, NodeType.segment, NodeType.statics, NodeType.save,
                         NodeType.channel_choose]:
            self.remove_btn.setEnabled(True)
        else:
            self.remove_btn.setEnabled(False)
        if node_type == NodeType.mask or node_type == NodeType.file_mask:
            self.mask_allow = False
            self.segment_allow = True
            self.file_mask_allow = False
            self.node_name = self.calculation_plan.get_node().operation.name
        elif node_type == NodeType.segment:
            self.mask_allow = True
            self.segment_allow = False
            self.file_mask_allow = False
            self.project_save_btn.setEnabled(True)
            # self.cmap_save_btn.setEnabled(True)
        elif node_type == NodeType.root or node_type == NodeType.channel_choose:
            self.mask_allow = False
            self.segment_allow = True
            self.file_mask_allow = True
            self.project_segmentation.setEnabled(True)
            self.choose_channel_btn.setEnabled(node_type == NodeType.root)
        elif node_type == NodeType.none or node_type == NodeType.statics or node_type == NodeType.save:
            self.mask_allow = False
            self.segment_allow = False
            self.file_mask_allow = False
        self.save_activate()
        self.plan_node_changed.emit()

    def mask_intersect(self):
        dial = TwoMaskDialog(self.mask_set)
        if dial.exec_():
            mask_name = str(self.mask_name.text()).strip()
            name1, name2 = dial.get_result()
            if self.update_element_btn.isChecked():
                self.calculation_plan.replace_step(MaskIntersection(mask_name, name1, name2))
            else:
                self.calculation_plan.add_step(MaskIntersection(mask_name, name1, name2))
            self.plan.update_view()

    def mask_sum(self):
        dial = TwoMaskDialog(self.mask_set)
        if dial.exec_():
            mask_name = str(self.mask_name.text()).strip()
            name1, name2 = dial.get_result()
            if self.update_element_btn.isChecked():
                self.calculation_plan.replace_step(MaskSum(mask_name, name1, name2))
            else:
                self.calculation_plan.add_step(MaskSum(mask_name, name1, name2))
            self.plan.update_view()

    def choose_channel(self):
        chanel_pos = self.chanel_pos.value()
        chanel_num = self.chanel_num.value()
        self.channels_used = True
        if self.update_element_btn.isChecked():
            self.calculation_plan.replace_step(ChooseChanel(chanel_pos, chanel_num))
        else:
            self.calculation_plan.add_step(ChooseChanel(chanel_pos, chanel_num))
        self.plan.update_view()

    def set_mask_name(self):
        name = str(self.mask_name.text()).strip()
        self.calculation_plan.replace_name(name)
        self.plan.update_view()

    def save_to_project(self):
        if self.save_constructor == CmapProfile:
            self.save_to_cmap()
            return
        if self.director_save_chk.isChecked():
            suffix, ok = QInputDialog.getText(self, "File name", "Set file name (without suffix)")
        else:
            suffix, ok = QInputDialog.getText(self, "File suffix", "Set file name suffix")
        if ok:
            suffix = str(suffix)
            if self.update_element_btn.isChecked():
                self.calculation_plan.replace_step(self.save_constructor(suffix, self.director_save_chk.isChecked()))
            else:
                self.calculation_plan.add_step(self.save_constructor(suffix, self.director_save_chk.isChecked()))
            self.plan.update_view()

    def save_to_cmap(self):
        dial = CmapSavePrepare("Settings for cmap create", self.director_save_chk.isChecked())
        if dial.exec_():
            if self.update_element_btn.isChecked():
                self.calculation_plan.replace_step(dial.get_result())
            else:
                self.calculation_plan.add_step(dial.get_result())
            self.plan.update_view()

    def save_mask(self):
        suffix, ok = QInputDialog.getText(self, "Mask file suffix", "Set mask name suffix")
        if ok:
            suffix = str(suffix)
            if self.update_element_btn.isChecked():
                self.calculation_plan.replace_step(MaskSave(suffix))
            else:
                self.calculation_plan.add_step(MaskSave(suffix))
            self.plan.update_view()

    def mask_by_mapping(self):
        name = str(self.mask_name.text()).strip()
        if self.update_element_btn.isChecked():
            node = self.calculation_plan.get_node()
            old_name = node.operation.name
            self.mask_set.remove(old_name)
            self.calculation_plan.replace_step(MaskFile(name, ""))
        else:
            self.calculation_plan.add_step(MaskFile(name, ""))
        self.plan.update_view()
        self.mask_set.add(name)
        self.mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def mask_by_suffix(self):
        name = str(self.mask_name.text()).strip()
        suffix = str(self.base_mask_name.text()).strip()
        if self.update_element_btn.isChecked():
            node = self.calculation_plan.get_node()
            old_name = node.operation.name
            self.mask_set.remove(old_name)
            self.calculation_plan.replace_step(MaskSuffix(name, suffix))
        else:
            self.calculation_plan.add_step(MaskSuffix(name, suffix))
        self.plan.update_view()
        self.mask_set.add(name)
        self.mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def mask_by_substitution(self):
        name = str(self.mask_name.text()).strip()
        base = str(self.base_mask_name.text()).strip()
        repl = str(self.swap_mask_name.text()).strip()
        if self.update_element_btn.isChecked():
            node = self.calculation_plan.get_node()
            old_name = node.operation.name
            self.mask_set.remove(old_name)
            self.calculation_plan.replace_step(MaskSub(name, base, repl))
        else:
            self.calculation_plan.add_step(MaskSub(name, base, repl))
        self.plan.update_view()
        self.mask_set.add(name)
        self.mask_text_changed()
        self.mask_name_changed(self.mask_name.text)

    def forgot_mask(self):
        self.calculation_plan.add_step(Operations.clean_mask)
        self.plan.update_view()

    def create_mask(self):
        text = str(self.mask_name.text()).strip()
        radius = self.dilate_radius_spin.value()
        if text != "" and text in self.mask_set:
            QMessageBox.warning(self, "Already exists", "Mask with this name already exists", QMessageBox.Ok)
            return
        if self.update_element_btn.isChecked():
            node = self.calculation_plan.get_node()
            name = node.operation.name
            self.mask_set.remove(name)
            self.mask_set.add(text)
            self.calculation_plan.replace_step(MaskCreate(text, radius))
            pass
        else:
            self.mask_set.add(text)
            self.calculation_plan.add_step(MaskCreate(text, radius))
        self.plan.update_view()
        self.mask_text_changed()

    def use_mask(self):
        text = str(self.mask_name.text())
        if text not in self.mask_set:
            QMessageBox.warning(self, "Don`t exists", "Mask with this name do not exists", QMessageBox.Ok)
            return
        self.calculation_plan.add_step(MaskUse(text))
        self.plan.update_view()

    def mask_name_changed(self, text):
        if str(text) in self.mask_set:
            self.generate_mask.setDisabled(True)
            self.reuse_mask.setDisabled(False)
        else:
            self.generate_mask.setDisabled(False)
            self.reuse_mask.setDisabled(True)

    def add_leave_biggest(self):
        profile = self.calculation_plan.get_node().operation
        profile.leave_biggest_swap()
        self.calculation_plan.replace_step(profile)
        self.plan.update_view()

    def add_segmentation(self):
        print("Buka")
        if self.channels_used and self.node_type == NodeType.root:
            ret = QMessageBox.question(self, "Segmentation from root",
                                        "You use channel choose in your plan. Are you sure "
                                        "to choose segmentation on root", QMessageBox.Cancel | QMessageBox.Ok,
                                        QMessageBox.Cancel)
            if ret == QMessageBox.Cancel:
                return

        text = str(self.segment_profile.currentItem().text())
        profile = self.settings.segmentation_profiles_dict[text]
        if self.update_element_btn.isChecked():
            self.calculation_plan.replace_step(profile)
        else:
            self.calculation_plan.add_step(profile)
        self.plan.update_view()

    def add_statistics(self):
        text = str(self.statistic_list.currentItem().text())
        statistics = self.settings.statistics_profile_dict[text]
        statistics_copy = deepcopy(statistics)
        prefix = str(self.statistic_name_prefix.text()).strip()
        statistics_copy.name_prefix = prefix
        if self.update_element_btn.isChecked():
            self.calculation_plan.replace_step(statistics_copy)
        else:
            self.calculation_plan.add_step(statistics_copy)
        self.plan.update_view()

    def remove_element(self):
        conflict_mask, used_mask = self.calculation_plan.get_file_mask_names()
        if len(conflict_mask) > 0:
            logging.info("Mask in use")
            QMessageBox.warning(self, "In use", "Masks {} are used in other places".format(", ".join(conflict_mask)))
            return
        self.mask_set -= used_mask
        self.calculation_plan.remove_step()
        self.plan.update_view()

    def clean_plan(self):
        self.channels_used = False
        self.calculation_plan = CalculationPlan()
        self.plan.set_plan(self.calculation_plan)
        self.node_type_changed()
        self.mask_set = set()

    def mask_text_changed(self):
        name = str(self.mask_name.text()).strip()
        self.suffix_mask_name_button.setDisabled(True)
        self.swap_mask_name_button.setDisabled(True)
        self.mapping_file_button.setDisabled(True)
        self.generate_mask.setDisabled(True)
        self.reuse_mask.setDisabled(True)
        self.set_mask_name.setDisabled(True)
        self.intersect_mask_btn.setDisabled(True)
        self.sum_mask_btn.setDisabled(True)
        # load mask from file
        if not self.update_element_btn.isChecked():
            self.set_mask_name.setDisabled(True)
            if self.file_mask_allow and (name == "" or name not in self.mask_set):
                base_text = str(self.base_mask_name.text()).strip()
                rep_text = str(self.swap_mask_name.text()).strip()
                self.suffix_mask_name_button.setEnabled(base_text != "")
                self.swap_mask_name_button.setEnabled((base_text != "") and (rep_text != ""))
                self.mapping_file_button.setEnabled(True)
                self.intersect_mask_btn.setEnabled(len(self.mask_set) > 1)
                self.sum_mask_btn.setEnabled(len(self.mask_set) > 1)
            # generate mask from segmentation
            if self.mask_allow and (name == "" or name not in self.mask_set):
                self.generate_mask.setEnabled(True)
            # reuse mask
            if self.file_mask_allow and name in self.mask_set:
                self.reuse_mask.setEnabled(True)
        # edit mask
        else:
            if self.node_type != NodeType.file_mask and self.node_type != NodeType.mask:
                return
            # change mask name
            if name not in self.mask_set and name != "":
                self.set_mask_name.setEnabled(True)
            if self.node_type == NodeType.file_mask and \
                    (name == "" or name == self.node_name or name not in self.mask_set):
                base_text = str(self.base_mask_name.text()).strip()
                rep_text = str(self.swap_mask_name.text()).strip()
                self.suffix_mask_name_button.setEnabled(base_text != "")
                self.swap_mask_name_button.setEnabled((base_text != "") and (rep_text != ""))
                self.mapping_file_button.setEnabled(True)
                self.intersect_mask_btn.setEnabled(len(self.mask_set) > 1)
                self.sum_mask_btn.setEnabled(len(self.mask_set) > 1)
            # generate mask from segmentation
            if self.node_type == NodeType.mask and (name == "" or name == self.node_name or name not in self.mask_set):
                self.generate_mask.setEnabled(True)
            # reuse mask
            if self.node_type == NodeType.file_mask and name in self.mask_set:
                self.reuse_mask.setEnabled(True)

    def add_calculation_plan(self, used_text=None):
        if used_text is None or isinstance(used_text, bool):
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title")
        else:
            text, ok = QInputDialog.getText(self, "Plan title", "Set plan title. Previous ({}) "
                                                                "is already in use".format(used_text))
        if ok:
            text = str(text)
            if text in self.settings.batch_plans:
                self.add_calculation_plan(text)
                return
            plan = copy(self.calculation_plan)
            plan.set_name(text)
            self.settings.batch_plans[text] = plan
            self.plan_created.emit()

    def showEvent(self, _):
        new_statistics = list(sorted(self.settings.statistics_profile_dict.keys()))
        new_segment = list(sorted(self.settings.segmentation_profiles_dict.keys()))
        if self.statistic_list.currentItem() is not None:
            text = str(self.statistic_list.currentItem().text())
            try:
                statistic_index = new_statistics.index(text)
            except ValueError:
                statistic_index = -1
        else:
            statistic_index = -1
        if self.segment_profile.currentItem() is not None:
            text = str(self.segment_profile.currentItem().text())
            try:
                segment_index = new_segment.index(text)
            except ValueError:
                segment_index = -1
        else:
            segment_index = -1
        self.protect = True
        self.statistic_list.clear()
        self.statistic_list.addItems(new_statistics)
        if statistic_index != -1:
            self.statistic_list.setCurrentRow(statistic_index)

        self.segment_profile.clear()
        self.segment_profile.addItems(new_segment)
        if segment_index != -1:
            self.segment_profile.setCurrentRow(segment_index)
        self.protect = False

    def show_statistics_info(self, text=None):
        if self.protect:
            return
        if text is None:
            if self.statistic_list.currentItem() is not None:
                text = str(self.statistic_list.currentItem().text())
            else:
                return
        self.information.setText(str(self.settings.statistics_profile_dict[str(text)]))

    def show_statistics(self):

        if self.update_element_btn.isChecked():
            if self.node_type == NodeType.statics:
                self.add_calculation.setEnabled(True)
            else:
                self.add_calculation.setDisabled(True)
        else:
            if self.statistic_list.currentItem() is not None:
                self.add_calculation.setEnabled(self.mask_allow)
            else:
                self.add_calculation.setDisabled(True)

    def show_segment_info(self, text=None):
        if self.protect:
            return
        if text is None:
            if self.segment_profile.currentItem() is not None:
                text = str(self.segment_profile.currentItem().text())
            else:
                return
        self.information.setText(str(self.settings.segmentation_profiles_dict[str(text)]))

    def show_segment(self):
        if self.update_element_btn.isChecked():
            self.get_big.setDisabled(True)
            if self.node_type == NodeType.segment:
                self.chose_profile.setEnabled(True)
            else:
                self.chose_profile.setDisabled(True)
        else:
            if self.node_type == NodeType.segment:
                self.get_big.setEnabled(True)
            else:
                self.get_big.setDisabled(True)
            if self.segment_profile.currentItem() is not None:
                self.chose_profile.setEnabled(self.segment_allow)
            else:
                self.chose_profile.setDisabled(True)

    def edit_plan(self):
        plan = self.sender().plan_to_edit  # type: CalculationPlan
        self.calculation_plan = copy(plan)
        self.channels_used = False
        for el in plan.execution_tree.children:
            if isinstance(el.operation, ChooseChanel):
                self.channels_used = True
        self.plan.set_plan(self.calculation_plan)
        self.mask_set.clear()
        self.calculation_plan.set_position([])
        self.mask_set.update(self.calculation_plan.get_mask_names())


class PlanPreview(QTreeWidget):
    """
    :type calculation_plan: CalculationPlan
    """
    changed_node = pyqtSignal()

    def __init__(self, parent=None, calculation_plan=None):
        super(PlanPreview, self).__init__(parent)
        self.calculation_plan = calculation_plan
        self.header().close()
        self.itemSelectionChanged.connect(self.set_path)

    def restore_path(self, widget, path):
        """
        :type widget: QTreeWidgetItem
        :type path: list[int]
        :param widget:
        :param path:
        :return:
        """
        if widget is None:
            return list(reversed(path))
        parent = widget.parent()
        if parent is None:
            return list(reversed(path))
        index = parent.indexOfChild(widget)
        if str(parent.child(0).text(0)) == "Description":
            index -= 1
        if index == -1:
            return None
        path.append(index)
        return self.restore_path(parent, path)

    def set_path(self):
        current_item = self.currentItem()  # type : QTreeWidgetItem
        if current_item is None:
            return
        self.calculation_plan.set_position(self.restore_path(current_item, []))
        self.changed_node.emit()

    def preview_object(self, calculation_plan):
        self.set_plan(calculation_plan)

    def set_plan(self, calculation_plan):
        self.calculation_plan = calculation_plan
        self.update_view(True)

    def explore_tree(self, up_widget, node_plan, deep=True):
        """
        :type up_widget: QTreeWidgetItem
        :type node_plan: CalculationTree
        :type deep: bool
        :param up_widget: List widget item
        :param node_plan: node from calculation plan
        :return:
        """
        widget = QTreeWidgetItem(up_widget)
        widget.setText(0, CalculationPlan.get_el_name(node_plan.operation))
        self.setCurrentItem(widget)
        if isinstance(node_plan.operation, StatisticProfile) or isinstance(node_plan.operation, SegmentationProfile):
            desc = QTreeWidgetItem(widget)
            desc.setText(0, "Description")
            description = str(node_plan.operation).split("\n")
            for line in description[1:]:
                if line.strip() == "":
                    continue
                w = QTreeWidgetItem(desc)
                w.setText(0, line)
        if deep:
            for el in node_plan.children:
                self.explore_tree(widget, el)
        up_widget.setExpanded(True)

    def get_node(self, path):
        """
        :type path: list[int]
        :param path:
        :return: QTreeWidgetItem
        """
        widget = self.topLevelItem(0)  # type : QTreeWidgetItem
        for index in path:
            if str(widget.child(0).text(0)) == "Description":
                index += 1
            widget = widget.child(index)
        return widget

    def update_view(self, reset=False):
        if reset:
            self.clear()
            root = QTreeWidgetItem(self)
            root.setText(0, "Root")
            self.setCurrentItem(root)
            for el in self.calculation_plan.execution_tree.children:
                self.explore_tree(root, el, True)
            return
        self.blockSignals(True)
        for i, (path, el, op_type) in enumerate(self.calculation_plan.get_changes()):
            if op_type == PlanChanges.add_node:
                node = self.get_node(path)
                self.explore_tree(node, el, False)
            elif op_type == PlanChanges.remove_node:
                node = self.get_node(path[:-1])
                index = path[-1]
                if str(node.child(0).text(0)) == "Description":
                    index += 1
                node.removeChild(node.child(index))
            elif op_type == PlanChanges.replace_node:
                node = self.get_node(path)
                node.setText(0, CalculationPlan.get_el_name(el.operation))
            else:
                logging.error("Unknown operation {}".format(op_type))
        self.blockSignals(False)
        self.set_path()
        self.changed_node.emit()


class CalculateInfo(QWidget):
    """
    :type settings: Settings
    """
    plan_to_edit_signal = pyqtSignal()

    def __init__(self, settings):
        super(CalculateInfo, self).__init__()
        self.settings = settings
        self.calculate_plans = QListWidget(self)
        self.plan_view = PlanPreview(self)
        self.delete_plan_btn = QPushButton("Delete plan")
        self.edit_plan_btn = QPushButton("Edit plan")
        self.export_plans_btn = QPushButton("Export plans")
        self.import_plans_btn = QPushButton("Import plans")
        info_layout = QVBoxLayout()
        info_butt_layout = QHBoxLayout()
        info_butt_layout.addWidget(self.delete_plan_btn)
        info_butt_layout.addWidget(self.edit_plan_btn)
        info_layout.addLayout(info_butt_layout)
        info_butt_layout = QHBoxLayout()
        info_butt_layout.addWidget(self.export_plans_btn)
        info_butt_layout.addWidget(self.import_plans_btn)
        info_layout.addLayout(info_butt_layout)
        info_chose_layout = QGridLayout()
        info_chose_layout.addWidget(QLabel("List of plans:"), 0, 0)
        info_chose_layout.addWidget(QLabel("Plan preview:"), 2, 0)
        info_chose_layout.addWidget(self.calculate_plans, 1, 0)
        info_chose_layout.addWidget(self.plan_view, 3, 0)
        info_layout.addLayout(info_chose_layout)
        self.setLayout(info_layout)
        self.calculate_plans.addItems(list(sorted(self.settings.batch_plans.keys())))
        self.protect = False
        self.plan_to_edit = None

        self.plan_view.header().close()
        self.calculate_plans.currentTextChanged.connect(self.plan_preview)
        self.delete_plan_btn.clicked.connect(self.delete_plan)
        self.edit_plan_btn.clicked.connect(self.edit_plan)
        self.export_plans_btn.clicked.connect(self.export_plans)
        self.import_plans_btn.clicked.connect(self.import_plans)

    def update_plan_list(self):
        new_plan_list = list(sorted(self.settings.batch_plans.keys()))
        if self.calculate_plans.currentItem() is not None:
            text = str(self.calculate_plans.currentItem().text())
            try:
                index = new_plan_list.index(text)
            except ValueError:
                index = -1
        else:
            index = -1
        self.protect = True
        self.calculate_plans.clear()
        self.calculate_plans.addItems(new_plan_list)
        if index != -1:
            self.calculate_plans.setCurrentRow(index)
        else:
            pass
            # self.plan_view.setText("")
        self.protect = False

    def export_plans(self):
        choose = ExportDialog(self.settings.batch_plans, PlanPreview)
        if not choose.exec_():
            return
        dial = QFileDialog(self, "Export calculation plans")
        dial.setFileMode(QFileDialog.AnyFile)
        dial.setAcceptMode(QFileDialog.AcceptSave)
        if self.settings.save_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setNameFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        dial.selectFile("calculation_plans.json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            self.settings.dump_calculation_plans(file_path, choose.get_checked())

    def import_plans(self):
        dial = QFileDialog(self, "Export calculation plans")
        dial.setFileMode(QFileDialog.ExistingFile)
        dial.setAcceptMode(QFileDialog.AcceptOpen)
        if self.settings.open_directory is not None:
            dial.setDirectory(self.settings.save_directory)
        dial.setNameFilter("Calculation plans (*.json)")
        dial.setDefaultSuffix("json")
        if dial.exec_():
            file_path = dial.selectedFiles()[0]
            plans = self.settings.load_calculation_plans(file_path)
            choose = ImportDialog(plans, self.settings.batch_plans, PlanPreview)
            if choose.exec_():
                self.settings.add_calculation_plans(plans, choose.get_import_list())
                self.update_plan_list()

    def delete_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if text == "":
            return
        if text in self.settings.batch_plans:
            del self.settings.batch_plans[text]
        self.update_plan_list()
        self.plan_view.clear()

    def edit_plan(self):
        if self.calculate_plans.currentItem() is None:
            return
        text = str(self.calculate_plans.currentItem().text())
        if text == "":
            return
        if text in self.settings.batch_plans:
            self.plan_to_edit = self.settings.batch_plans[text]
            self.plan_to_edit_signal.emit()

    def plan_preview(self, text):
        if self.protect:
            return
        text = str(text)
        if text.strip() == "":
            return
        plan = self.settings.batch_plans[str(text)]  # type: CalculationPlan
        self.plan_view.set_plan(plan)


class CalculatePlaner(QSplitter):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent):
        QWidget.__init__(self, parent)
        self.settings = settings
        self.info_widget = CalculateInfo(settings)
        self.addWidget(self.info_widget)
        self.create_plan = CreatePlan(settings)
        self.create_plan.plan_created.connect(self.info_widget.update_plan_list)
        self.info_widget.plan_to_edit_signal.connect(self.create_plan.edit_plan)
        self.addWidget(self.create_plan)


class CmapSavePrepare(QDialog):
    def __init__(self, text, directory, title="Cmap settings"):
        super(CmapSavePrepare, self).__init__()
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        self.directory = directory
        self.setWindowTitle(title)
        self.gauss_type = QComboBox(self)
        self.gauss_type.addItems(["No gauss", "2d gauss", "2d + 3d gauss"])
        self.center_data = QCheckBox(self)
        self.center_data.setChecked(True)
        # self.with_statistics = QCheckBox(self)
        # self.with_statistics.setChecked(True)
        self.rotation_axis = QComboBox(self)
        self.rotation_axis.addItems(["None", "x", "y", "z"])
        self.cut_data = QCheckBox(self)
        self.cut_data.setChecked(True)
        self.suffix_text = QLineEdit()
        grid = QGridLayout()
        grid.addWidget(QLabel("Gauss type"), 0, 0)
        grid.addWidget(self.gauss_type, 0, 1)
        grid.addWidget(QLabel("Center data"), 1, 0)
        grid.addWidget(self.center_data, 1, 1)
        # grid.addWidget(QLabel("With statistics"), 2, 0)
        # grid.addWidget(self.with_statistics, 2, 1)
        grid.addWidget(QLabel("Rotation axis"), 3, 0)
        grid.addWidget(self.rotation_axis, 3, 1)
        grid.addWidget(QLabel("Cut obsolete area"), 4, 0)
        grid.addWidget(self.cut_data, 4, 1)
        if directory:
            grid.addWidget(QLabel("File name"), 5, 0)
        else:
            grid.addWidget(QLabel("File suffix"), 5, 0)
        grid.addWidget(self.suffix_text, 5, 1)

        close = QPushButton("Cancel")
        close.clicked.connect(self.close)
        save = QPushButton("Accept")
        save.clicked.connect(self.save)

        button_layout = QHBoxLayout()
        button_layout.addWidget(close)
        button_layout.addStretch()
        button_layout.addWidget(save)

        layout = QVBoxLayout()
        layout.addWidget(text_label)
        layout.addLayout(grid)
        layout.addLayout(button_layout)

        self.setLayout(layout)

    def save(self):
        self.accept()

    def get_result(self):
        options = {"No gauss": GaussUse.no_gauss, "2d gauss": GaussUse.gauss_2d, "2d + 3d gauss": GaussUse.gauss_3d}
        return CmapProfile(suffix=str(self.suffix_text.text()), gauss_type=options[str(self.gauss_type.currentText())],
                           center_data=self.center_data.isChecked(), rotation_axis=str(self.rotation_axis.currentText())
                           , cut_obsolete_area=self.cut_data.isChecked(), directory=self.directory)
