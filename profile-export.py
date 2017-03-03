from qt_import import QDialog, QListWidget, QListWidgetItem, Qt, QPushButton, QHBoxLayout, QVBoxLayout
import numpy as np


class ExportDialog(QDialog):
    def __init__(self, export_dict, viewer):
        super(ExportDialog, self).__init__()
        self.setWindowTitle("Export")
        self.export_dict = export_dict
        self.viewer = viewer()
        self.list_view = QListWidget()
        self.check_state = np.zeros(len(export_dict), dtype=np.bool)
        self.check_state[...] = True
        for el in sorted(export_dict.keys()):
            item = QListWidgetItem(el)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.list_view.addItem(item)

        self.export_btn = QPushButton("Export")
        self.cancel_btn = QPushButton("Cancel")
        self.check_btn = QPushButton("Check all")
        self.uncheck_btn = QPushButton("Uncheck all")

        self.cancel_btn.clicked.connect(self.close)
        self.export_btn.clicked.connect(self.accept)
        self.check_btn.clicked.connect(self.check_all)
        self.uncheck_btn.clicked.connect(self.uncheck_all)

        self.list_view.itemSelectionChanged.connect()

        layout = QVBoxLayout()
        info_layout = QHBoxLayout()
        info_layout.addWidget(self.list_view)
        info_layout.addWidget(self.viewer)
        layout.addLayout(layout)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.check_btn)
        btn_layout.addWidget(self.uncheck_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(self.export_btn)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

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

    def check_all(self):
        for index in range(self.list_view.count()):
            item = self.list_view.item(index)
            item.setCheckState(Qt.Checked)
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
    def __init__(self):
        super(ImportDialog, self).__init__()
        self.setWindowTitle("Import")
