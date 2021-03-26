import re

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from PartSeg.common_gui.searchable_list_widget import SearchableListWidget
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict


class ExportDialog(QDialog):
    def __init__(self, export_dict, viewer):
        super().__init__()
        self.setWindowTitle("Export")
        self.export_dict = export_dict
        self.viewer = viewer()
        self.list_view = SearchableListWidget()
        self.check_state = np.zeros(len(export_dict), dtype=bool)
        self.check_state[...] = True
        for el in sorted(export_dict.keys()):
            item = QListWidgetItem(el)
            # noinspection PyTypeChecker
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_view.addItem(item)

        self.checked_num = len(export_dict)

        self.export_btn = QPushButton("Export")
        self.cancel_btn = QPushButton("Cancel")
        self.check_btn = QPushButton("Check all")
        self.uncheck_btn = QPushButton("Uncheck all")

        self.cancel_btn.clicked.connect(self.close)
        self.export_btn.clicked.connect(self.accept)
        self.check_btn.clicked.connect(self.check_all)
        self.uncheck_btn.clicked.connect(self.uncheck_all)

        self.list_view.itemSelectionChanged.connect(self.preview)
        self.list_view.itemChanged.connect(self.checked_change)

        layout = QVBoxLayout()
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.list_view)
        info_layout.addWidget(self.viewer)
        layout.addLayout(info_layout)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.check_btn)
        btn_layout.addWidget(self.uncheck_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def checked_change(self, item):
        if item.checkState() == Qt.Unchecked:
            self.checked_num -= 1
        else:
            self.checked_num += 1
        if self.checked_num == 0:
            self.export_btn.setDisabled(True)
        else:
            self.export_btn.setEnabled(True)

    def get_checked(self):
        res = []
        for i in range(self.list_view.count()):
            it = self.list_view.item(i)
            if it.checkState() == Qt.Checked:
                res.append(str(it.text()))
        return res

    def preview(self):
        name = str(self.list_view.currentItem().text())
        self.viewer.preview_object(self.export_dict[name])

    def check_change(self):
        item = self.list_view.currentItem()  # type: QListWidgetItem
        index = self.list_view.currentRow()
        checked = item.checkState() == Qt.Checked
        self.check_state[index] = checked
        self.export_btn.setEnabled(np.any(self.check_state))

    def uncheck_all(self):
        for index in range(self.list_view.count()):
            item = self.list_view.item(index)
            item.setCheckState(Qt.Unchecked)
        self.check_state[...] = False
        self.export_btn.setDisabled(True)
        self.checked_num = 0

    def check_all(self):
        for index in range(self.list_view.count()):
            item = self.list_view.item(index)
            item.setCheckState(Qt.Checked)
        self.checked_num = len(self.export_dict)
        self.check_state[...] = True
        self.export_btn.setDisabled(False)

    def get_export_list(self):
        res = []
        for num in range(self.list_view.count()):
            item = self.list_view.item(num)
            if item.checkState() == Qt.Checked:
                res.append(str(item.text()))
        return res


class ImportDialog(QDialog):
    def __init__(self, import_dict, local_dict, viewer):
        """
        :type import_dict: dict[str, object]
        :type local_dict: dict[str, object]
        :param import_dict:
        :param local_dict:
        :param viewer:
        """
        super().__init__()
        self.setWindowTitle("Import")
        self.viewer = viewer()
        self.local_viewer = viewer()
        self.import_dict = import_dict
        self.local_dict = local_dict
        conflicts = set(local_dict.keys()) & set(import_dict.keys())
        # print(conflicts)

        self.list_view = QTreeWidget()
        self.list_view.setColumnCount(4)
        self.radio_group_list = []
        self.checked_num = len(import_dict)

        def rename_func(ob_name, new_name_field, rename_radio):
            end_reg = re.compile(r"(.*) \((\d+)\)$")

            def in_func():
                if not rename_radio.isChecked() or str(new_name_field.text()).strip() != "":
                    return

                match = end_reg.match(ob_name)
                if match:
                    new_name_format = match.group(1) + " ({})"
                    i = int(match.group(2)) + 1
                else:
                    new_name_format = ob_name + " ({})"
                    i = 1
                while new_name_format.format(i) in self.local_dict:
                    i += 1
                new_name_field.setText(new_name_format.format(i))

            return in_func

        def block_import(radio_btn, name_field):
            def inner_func():
                text = str(name_field.text()).strip()
                if text == "" and radio_btn.isChecked():
                    self.import_btn.setDisabled(True)
                else:
                    self.import_btn.setEnabled(True)

            return inner_func

        for name in sorted(import_dict.keys()):
            item = QTreeWidgetItem()
            item.setText(0, name)
            # noinspection PyTypeChecker
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked)
            self.list_view.addTopLevelItem(item)
            if name in conflicts:
                group = QButtonGroup()
                overwrite = QRadioButton("Overwrite")
                overwrite.setChecked(True)
                rename = QRadioButton("Rename")
                new_name = QLineEdit()
                new_name.textChanged.connect(block_import(rename, new_name))
                rename.toggled.connect(block_import(rename, new_name))
                overwrite.toggled.connect(block_import(rename, new_name))

                rename.toggled.connect(rename_func(name, new_name, rename))
                group.addButton(overwrite)
                group.addButton(rename)
                self.radio_group_list.append(group)
                self.list_view.setItemWidget(item, 1, overwrite)
                self.list_view.setItemWidget(item, 2, rename)
                self.list_view.setItemWidget(item, 3, new_name)

        self.import_btn = QPushButton("Import")
        self.cancel_btn = QPushButton("Cancel")
        self.check_btn = QPushButton("Check all")
        self.uncheck_btn = QPushButton("Uncheck all")

        self.cancel_btn.clicked.connect(self.close)
        self.import_btn.clicked.connect(self.accept)
        self.check_btn.clicked.connect(self.check_all)
        self.uncheck_btn.clicked.connect(self.uncheck_all)

        self.list_view.itemSelectionChanged.connect(self.preview)
        self.list_view.itemChanged.connect(self.checked_change)

        layout = QVBoxLayout()
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.list_view, 2)
        v1_lay = QVBoxLayout()
        v1_lay.addWidget(QLabel("Import:"))
        v1_lay.addWidget(self.viewer)
        info_layout.addLayout(v1_lay, 1)
        # info_layout.addWidget(self.local_viewer, 1)
        v2_lay = QVBoxLayout()
        v2_lay.addWidget(QLabel("Local:"))
        v2_lay.addWidget(self.local_viewer)
        info_layout.addLayout(v2_lay, 1)
        layout.addLayout(info_layout)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.check_btn)
        btn_layout.addWidget(self.uncheck_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.import_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def preview(self):
        item = self.list_view.currentItem()
        name = str(item.text(0))
        self.viewer.preview_object(self.import_dict[name])
        if self.list_view.itemWidget(item, 1) is not None:
            self.local_viewer.preview_object(self.local_dict[name])
        else:
            self.local_viewer.clear()

    def checked_change(self, item, _):
        if item.checkState(0) == Qt.Unchecked:
            self.checked_num -= 1
            if self.list_view.itemWidget(item, 1) is not None:
                self.list_view.itemWidget(item, 1).setDisabled(True)
                self.list_view.itemWidget(item, 2).setDisabled(True)
                self.list_view.itemWidget(item, 3).setDisabled(True)
        else:
            self.checked_num += 1
            if self.list_view.itemWidget(item, 1) is not None:
                self.list_view.itemWidget(item, 1).setEnabled(True)
                self.list_view.itemWidget(item, 2).setEnabled(True)
                self.list_view.itemWidget(item, 3).setEnabled(True)
        if self.checked_num == 0:
            self.import_btn.setDisabled(True)
        else:
            self.import_btn.setEnabled(True)

    def get_import_list(self):
        res = []
        for index in range(self.list_view.topLevelItemCount()):
            item = self.list_view.topLevelItem(index)
            if item.checkState(0) == Qt.Checked:
                chk = self.list_view.itemWidget(item, 2)
                if chk is not None and chk.isChecked():
                    res.append((str(item.text(0)), str(self.list_view.itemWidget(item, 3).text())))
                else:
                    name = str(item.text(0))
                    res.append((name, name))
        return res

    def uncheck_all(self):
        for index in range(self.list_view.topLevelItemCount()):
            item = self.list_view.topLevelItem(index)
            item.setCheckState(0, Qt.Unchecked)
        self.check_state[...] = False
        self.import_btn.setDisabled(True)
        self.checked_num = 0

    def check_all(self):
        for index in range(self.list_view.topLevelItemCount()):
            item = self.list_view.topLevelItem(index)
            item.setCheckState(0, Qt.Checked)
        self.checked_num = len(self.import_dict)
        self.check_state[...] = True
        self.import_btn.setDisabled(False)


class ObjectPreview(QTextEdit):
    """Base class for viewer used by :py:class:`ExportDialog` to preview data  """

    def preview_object(self, ob):
        raise NotImplementedError()


class StringViewer(ObjectPreview):
    """Simple __str__ serialization"""

    def preview_object(self, ob):
        self.setText(str(ob))


class ProfileDictViewer(ObjectPreview):
    """
    Preview of :py:class"`SegmentationProfile`.
    Serialized using :py:meth:`ObjectPreview.pretty_print`.
    """

    def preview_object(self, ob: ROIExtractionProfile):
        text = ob.pretty_print(analysis_algorithm_dict)
        self.setText(text)
