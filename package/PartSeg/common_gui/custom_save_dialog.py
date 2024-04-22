import os
import typing
from contextlib import suppress
from pathlib import Path

from qtpy.QtWidgets import QDialog, QFileDialog, QGridLayout, QPushButton, QStackedWidget

from PartSeg.common_gui.algorithms_description import FormWidget
from PartSeg.common_gui.custom_load_dialog import IORegister, LoadRegisterFileDialog
from PartSegCore.algorithm_describe_base import get_fields_from_algorithm
from PartSegCore.io_utils import SaveBase

if typing.TYPE_CHECKING:  # pragma: no cover
    from PartSeg.common_backend.base_settings import BaseSettings


class SaveProperty(typing.NamedTuple):
    save_destination: typing.Union[str, typing.List[str]]
    selected_filter: str
    save_class: SaveBase
    parameters: dict


class FormDialog(QDialog):
    @staticmethod
    def widget_class() -> typing.Type[FormWidget]:
        return FormWidget

    def __init__(self, fields, values=None, image=None, settings=None, parent=None):
        super().__init__(parent)
        self.widget = self.widget_class()(fields, settings=settings)
        if values is not None:
            self.widget.set_values(values)
        if image is not None:
            self.widget.image_changed(image)
        self.accept_btn = QPushButton("Save")
        self.accept_btn.clicked.connect(self.accept)
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.clicked.connect(self.reject)
        layout = QGridLayout()
        layout.addWidget(self.widget, 0, 0, 1, 2)
        layout.addWidget(self.reject_btn, 1, 0)
        layout.addWidget(self.accept_btn, 1, 1)
        self.setLayout(layout)

    def get_values(self):
        return self.widget.get_values()

    def set_values(self, values):
        return self.widget.set_values(values)


class CustomSaveDialog(LoadRegisterFileDialog):
    def __init__(
        self,
        save_register: IORegister,
        system_widget=True,
        base_values: typing.Optional[dict] = None,
        parent=None,
        caption="Save file",
        history: typing.Optional[typing.List[str]] = None,
        file_mode=QFileDialog.FileMode.AnyFile,
    ):
        super().__init__(save_register, caption, parent)
        self.setFileMode(file_mode)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, not system_widget)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        self.filterSelected.connect(self.change_filter)
        self.accepted_native = False
        self.values = {}
        self.names = []
        if history is not None:
            history = self.history() + history
            self.setHistory(history)
        self.base_values = base_values if base_values is not None else {}
        if not system_widget:
            widget = QStackedWidget()
            for name, val in self.io_register.items():
                wi = FormWidget(val)
                if name in self.base_values:
                    wi.set_values(self.base_values[name])
                widget.addWidget(wi)
                self.names.append(name)

            self.filterSelected.connect(self.change_parameters)

            layout = self.layout()
            if isinstance(layout, QGridLayout):
                # noinspection PyArgumentList
                layout.addWidget(widget, 0, layout.columnCount(), layout.rowCount(), 1)
            else:
                layout.addWidget(widget)
            self.stack_widget = widget
            self.selectNameFilter(self.names[0])

    def change_parameters(self, text):
        if not hasattr(self, "stack_widget"):
            return
        with suppress(ValueError):
            self.stack_widget.setCurrentIndex(self.names.index(text))
            if typing.cast(FormWidget, self.stack_widget.currentWidget()).has_elements():
                self.stack_widget.show()
            else:
                self.stack_widget.hide()

    def selectNameFilter(self, filter_name: str):
        with suppress(IndexError):
            self.change_parameters(filter_name)
        super().selectNameFilter(filter_name)
        with suppress(KeyError):
            ext = self.io_register[filter_name].get_default_extension()
            self.setDefaultSuffix(ext)

    def change_filter(self, current_filter):
        if current_filter not in self.io_register:
            return
        ext = self.io_register[current_filter].get_default_extension()
        self.setDefaultSuffix(ext)

    def accept(self):
        self.accepted_native = True
        if hasattr(self, "stack_widget"):
            self.values = typing.cast(FormWidget, self.stack_widget.currentWidget()).get_values()
            super().accept()
            return
        save_class = self.io_register[self.selectedNameFilter()]
        fields = get_fields_from_algorithm(save_class)
        if len(fields) == 0:
            super().accept()
            return
        dial = FormDialog(fields)
        if self.selectedNameFilter() in self.base_values:
            dial.set_values(self.base_values[self.selectedNameFilter()])
        if dial.exec_():
            self.values = dial.get_values()
            super().accept()
        else:
            super().reject()

    def get_result(self) -> SaveProperty:
        files = self.selectedFiles()
        return SaveProperty(
            files[0] if len(files) == 1 else files,
            self.selectedNameFilter(),
            self.io_register[self.selectedNameFilter()],
            self.values,
        )


class PSaveDialog(CustomSaveDialog):
    def __init__(
        self,
        save_register: IORegister,
        *,
        settings: "BaseSettings",
        path: str,
        default_directory: typing.Optional[str] = None,
        filter_path="",
        system_widget=True,
        base_values: typing.Optional[dict] = None,
        parent=None,
        caption="Save file",
        file_mode=QFileDialog.FileMode.AnyFile,
    ):
        if default_directory is None:
            default_directory = str(Path.home())
        super().__init__(
            save_register=save_register,
            system_widget=system_widget,
            base_values=base_values,
            parent=parent,
            caption=caption,
            history=settings.get_path_history(),
            file_mode=file_mode,
        )
        self.settings = settings
        self.path_in_dict = path
        self.filter_path = filter_path
        self.setDirectory(self.settings.get(path, default_directory))
        if self.filter_path:
            self.selectNameFilter(self.settings.get(self.filter_path, ""))

    def accept(self):
        super().accept()
        if self.result() != QDialog.DialogCode.Accepted:
            return
        directory = os.path.dirname(self.selectedFiles()[0])
        self.settings.add_path_history(directory)
        self.settings.set(self.path_in_dict, directory)
        if self.filter_path:
            self.settings.set(self.filter_path, self.selectedNameFilter())


SaveDialog = CustomSaveDialog
