from qtpy.QtWidgets import QFileDialog
import typing
from os.path import basename
from PartSeg.utils.io_utils import LoadBase


class LoadProperty(typing.NamedTuple):
    load_location: typing.List[str]
    selected_filter: str
    load_class: LoadBase


class CustomLoadDialog(QFileDialog):
    def __init__(self, load_register: typing.Dict[str, type(LoadBase)], parent=None,
                 history: typing.Optional[typing.List[str]] = None):
        super().__init__(parent)
        self.load_register = dict((x.get_name_with_suffix(), x) for x in load_register.values())
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
        self.files_list.extend(self.selectedFiles())
        chosen_class: LoadBase = self.load_register[self.selectedNameFilter()]
        if len(self.files_list) < chosen_class.number_of_files():
            self.setNameFilters([chosen_class.get_name()])
            self.setWindowTitle("Open File for:" + ",".join([basename(x) for x in self.files_list]))
            self.selectFile(chosen_class.get_next_file(self.files_list))
        else:
            super().accept()

    def get_result(self):
        chosen_class: LoadBase = self.load_register[self.selectedNameFilter()]
        return LoadProperty(self.files_list, self.selectedNameFilter(), chosen_class)
