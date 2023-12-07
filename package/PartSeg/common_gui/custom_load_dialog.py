import typing
from os.path import basename, dirname, isdir, isfile
from pathlib import Path

from qtpy.QtWidgets import QFileDialog

from PartSegCore.io_utils import LoadBase

if typing.TYPE_CHECKING:  # pragma: no cover
    from PartSeg.common_backend.base_settings import BaseSettings


class LoadProperty(typing.NamedTuple):
    load_location: typing.List[typing.Union[str, Path]]
    selected_filter: str
    load_class: typing.Type[LoadBase]


IORegister = typing.Union[typing.Dict[str, type(LoadBase)], type(LoadBase), str, typing.List[type(LoadBase)]]


class IOMethodMock:
    __new_style__ = False

    def __init__(self, name: str):
        self.name = name

    def get_name(self) -> str:
        return self.name

    def get_name_with_suffix(self) -> str:
        return self.get_name()

    def get_short_name(self) -> str:
        return self.get_name()

    @staticmethod
    def get_default_extension() -> str:
        return ""

    @staticmethod
    def need_segmentation() -> bool:
        return False

    @staticmethod
    def need_mask() -> bool:
        return False

    @staticmethod
    def get_fields() -> list:
        return []

    @staticmethod
    def number_of_files() -> int:
        return 1

    @staticmethod
    def save(*args, **kwargs):
        """For keep compatibility with SaveBase"""


class LoadRegisterFileDialog(QFileDialog):
    def __init__(
        self,
        io_register: IORegister,
        caption,
        parent=None,
    ):
        if isinstance(io_register, str):
            io_register = {io_register: IOMethodMock(io_register)}
        if isinstance(io_register, list):
            io_register = {x.get_name(): x for x in io_register}
        if not isinstance(io_register, typing.MutableMapping):
            io_register = {io_register.get_name(): io_register}
        super().__init__(parent, caption)
        self.io_register = {x.get_name_with_suffix(): x for x in io_register.values()}
        self.setNameFilters(list(self.io_register.keys()))


class CustomLoadDialog(LoadRegisterFileDialog):
    def __init__(
        self,
        load_register: IORegister,
        parent=None,
        caption="Load file",
        history: typing.Optional[typing.List[str]] = None,
    ):
        super().__init__(load_register, caption, parent)
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        self.files_list = []
        self.setWindowTitle("Open File")
        if history is not None:
            history = self.history() + history
            self.setHistory(history)

    def accept(self):
        selected_files = [
            x for x in self.selectedFiles() if self.fileMode == QFileDialog.FileMode.Directory or isfile(x)
        ]
        if not selected_files:
            return
        if len(selected_files) == 1 and self.fileMode != QFileDialog.FileMode.Directory and isdir(selected_files[0]):
            super().accept()
            return

        self.files_list.extend(selected_files)
        chosen_class: LoadBase = self.io_register[self.selectedNameFilter()]
        if len(self.files_list) < chosen_class.number_of_files():
            self.setNameFilters([chosen_class.get_name()])
            self.setWindowTitle("Open File for:" + ",".join(basename(x) for x in self.files_list))

            self.selectFile(chosen_class.get_next_file(self.files_list))
        else:
            super().accept()

    def get_result(self) -> LoadProperty:
        chosen_class: typing.Type[LoadBase] = self.io_register[self.selectedNameFilter()]
        return LoadProperty(self.files_list, self.selectedNameFilter(), chosen_class)


class PLoadDialog(CustomLoadDialog):
    def __init__(
        self,
        load_register: typing.Union[typing.Dict[str, type(LoadBase)], type(LoadBase)],
        *,
        settings: "BaseSettings",
        path: str,
        default_directory: typing.Optional[str] = None,
        filter_path="",
        parent=None,
        caption="Load file",
    ):
        if default_directory is None:
            default_directory = str(Path.home())
        super().__init__(
            load_register=load_register,
            parent=parent,
            caption=caption,
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
        if self.result() != QFileDialog.DialogCode.Accepted:
            return
        directory = dirname(self.selectedFiles()[0])
        self.settings.add_path_history(directory)
        self.settings.set(self.path_in_dict, directory)
        if self.filter_path:
            self.settings.set(self.filter_path, self.selectedNameFilter())


class SelectDirectoryDialog(QFileDialog):
    def __init__(
        self,
        settings: "BaseSettings",
        settings_path: typing.Union[str, typing.List[str]],
        default_directory: typing.Optional[str] = None,
        parent=None,
    ) -> None:
        super().__init__(parent, "Select directory")
        if default_directory is None:
            default_directory = str(Path.home())
        self.settings = settings
        self.setFileMode(QFileDialog.FileMode.Directory)
        self.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
        if isinstance(settings_path, list):
            for path_ in reversed(settings_path):
                default_directory = self.settings.get(path_, default_directory)
            self.setDirectory(default_directory)
            self.path_in_dict = settings_path[0]
        else:
            self.setDirectory(self.settings.get(settings_path, default_directory))
            self.path_in_dict = settings_path
        history = self.history() + settings.get_path_history()
        self.setHistory(history)

    def accept(self) -> None:
        super().accept()
        if self.result() != QFileDialog.DialogCode.Accepted:
            return
        directory = self.selectedFiles()[0]
        self.settings.add_path_history(directory)
        self.settings.set(self.path_in_dict, directory)
