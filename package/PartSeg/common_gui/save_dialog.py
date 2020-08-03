from enum import Enum

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from PartSegCore.channel_class import Channel
from PartSegCore.class_generator import BaseSerializableClass


class GetPropertyWidget(QWidget):
    def get_values(self) -> BaseSerializableClass:
        raise NotImplementedError


class SavePropertyDialog(QDialog):
    def __init__(self, description: str, widget: GetPropertyWidget):
        super().__init__(self)
        self.widget = widget
        save = QPushButton("Save", self)
        save.clicked.connect(self.accept)
        cancel = QPushButton("Cancel", self)
        cancel.clicked.connect(self.reject)
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel(description))
        layout.addWidget(self.widget)
        btn_layout = QVBoxLayout()
        btn_layout.addWidget(cancel)
        btn_layout.addWidget(cancel)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_values(self):
        return self.widget.get_values()


class GenericGetPropertyWidget(GetPropertyWidget):
    def __init__(self, properties: BaseSerializableClass, parent=None):
        super().__init__(parent)
        self.properties = properties
        self.widget_list = []
        layout = QFormLayout()
        for key, val in properties.__annotations__.items():
            name = key.replace("_", " ").capitalize()
            widget = self.get_widget(val)
            layout.addRow(name, widget)
            self.widget_list.append((key, widget))
        self.setLayout(layout)

    def get_values(self):
        resp = []
        for key, widget in self.widget_list:
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                resp.append((key, widget.value()))
            elif isinstance(widget, QLineEdit):
                resp.append((key, widget.text()))
            elif isinstance(widget, QCheckBox):
                resp.append((key, widget.isChecked()))
            elif isinstance(widget, QComboBox):
                type_ = self.properties.__annotations__[key]
                if issubclass(type_, Enum):
                    resp.append((key, list(type_.__members__.values())[widget.currentIndex()]))
                elif issubclass(type_, Channel):
                    resp.append((key, Channel(widget.currentIndex())))

        return dict(resp)

    @staticmethod
    def get_widget(type_):
        if issubclass(type_, Channel):
            res = QComboBox()
            res.addItems([str(x) for x in range(10)])
            return res
        if issubclass(type_, bool):
            return QCheckBox()
        if issubclass(type_, int):
            return QSpinBox()
        if issubclass(type_, float):
            return QDoubleSpinBox()
        if issubclass(type_, str):
            return QLineEdit()
        if issubclass(type_, Enum):
            res = QComboBox()
            res.addItems(list(type_.__members__.keys()))
            return res
