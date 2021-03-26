import typing
from os.path import basename, isdir, isfile

from qtpy.QtWidgets import QFileDialog

from PartSegCore.io_utils import LoadBase


class LoadProperty(typing.NamedTuple):
    load_location: typing.List[str]
    selected_filter: str
    load_class: LoadBase


class CustomLoadDialog(QFileDialog):
    def __init__(
        self,
        load_register: typing.Dict[str, type(LoadBase)],
        parent=None,
        history: typing.Optional[typing.List[str]] = None,
    ):
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
        selected_files = self.selectedFiles()
        if len(selected_files) == 0:
            return
        if len(selected_files) == 1 and self.fileMode != QFileDialog.Directory and isdir(selected_files[0]):
            super().accept()
            return

        self.files_list.extend([x for x in self.selectedFiles() if self.fileMode == QFileDialog.Directory or isfile(x)])
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
