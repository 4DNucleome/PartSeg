import typing
from os.path import basename, dirname, isdir, isfile
from pathlib import Path

from qtpy.QtWidgets import QFileDialog

from PartSegCore.io_utils import LoadBase

if typing.TYPE_CHECKING:  # pragma: no cover
    from PartSeg.common_backend.base_settings import BaseSettings


class LoadProperty(typing.NamedTuple):
    load_location: typing.List[str]
    selected_filter: str
    load_class: LoadBase


class CustomLoadDialog(QFileDialog):
    def __init__(
        self,
        load_register: typing.Union[typing.Dict[str, type(LoadBase)], type(LoadBase)],
        parent=None,
        history: typing.Optional[typing.List[str]] = None,
    ):
        if not isinstance(load_register, dict):
            load_register = {load_register.get_name(): load_register}
        super().__init__(parent)
        self.load_register = {x.get_name_with_suffix(): x for x in load_register.values()}
        self.setOption(QFileDialog.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.ExistingFile)
        self.setAcceptMode(QFileDialog.AcceptOpen)
        self.setNameFilters(self.load_register.keys())
        self.files_list = []
        self.setWindowTitle("Open File")
        if history is not None:
            history = self.history() + history
            self.setHistory(history)

    def accept(self):
        selected_files = [x for x in self.selectedFiles() if self.fileMode == QFileDialog.Directory or isfile(x)]
        if not selected_files:
            return
        if len(selected_files) == 1 and self.fileMode != QFileDialog.Directory and isdir(selected_files[0]):
            super().accept()
            return

        self.files_list.extend(selected_files)
        chosen_class: LoadBase = self.load_register[self.selectedNameFilter()]
        if len(self.files_list) < chosen_class.number_of_files():
            self.setNameFilters([chosen_class.get_name()])
            self.setWindowTitle("Open File for:" + ",".join(basename(x) for x in self.files_list))

            self.selectFile(chosen_class.get_next_file(self.files_list))
        else:
            super().accept()

    def get_result(self):
        chosen_class: LoadBase = self.load_register[self.selectedNameFilter()]
        return LoadProperty(self.files_list, self.selectedNameFilter(), chosen_class)


class PLoadDialog(CustomLoadDialog):
    def __init__(
        self,
        load_register: typing.Union[typing.Dict[str, type(LoadBase)], type(LoadBase)],
        *,
        settings: "BaseSettings",
        path: str,
        default_directory=str(Path.home()),
        filter_path="",
        parent=None,
    ):
        super().__init__(
            load_register=load_register,
            parent=parent,
            history=settings.get_path_history(),
        )
        self.settings = settings
        self.path_in_dict = path
        self.filter_path = filter_path
        self.setDirectory(self.settings.get(path, default_directory))
        if self.filter_path:
            self.selectNameFilter(self.settings.get(self.filter_path, ""))

    def accept(self):
        super().accept()
        if self.result() != QFileDialog.Accepted:
            return
        directory = dirname(self.selectedFiles()[0])
        self.settings.add_path_history(directory)
        self.settings.set(self.path_in_dict, directory)
        if self.filter_path:
            self.settings.set(self.filter_path, self.selectedNameFilter())
