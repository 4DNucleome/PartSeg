"""
Implement dictionary viewer widget
"""

from __future__ import annotations

from qtpy.QtWidgets import QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget


class DictViewer(QWidget):
    def __init__(self, data: dict | None = None):
        super().__init__()
        self._data = {}
        self.tree = QTreeWidget()
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tree)
        self.setLayout(self.layout)
        self.tree.setHeaderLabels(["Key", "Value"])
        self.set_data(data)

    def set_data(self, data: dict | None):
        if data is None:
            data = {}
        self._data = data
        self.tree.clear()
        self.fill_tree(data, self.tree)

    def fill_tree(self, data: dict, parent: QTreeWidget | QTreeWidgetItem):
        for key, value in data.items():
            if isinstance(value, dict):
                key_item = QTreeWidgetItem(parent, [key])
                self.fill_tree(value, key_item)
            elif isinstance(value, list):
                key_item = QTreeWidgetItem(parent, [key])
                for i, val in enumerate(value):
                    self.fill_tree({str(i): val}, key_item)
            else:
                QTreeWidgetItem(parent, [key, str(value)])
