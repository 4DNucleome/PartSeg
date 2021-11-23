# pylint: disable=R0201
import os
import platform
import sys
from enum import Enum
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import qtpy
from qtpy.QtCore import QSize, Qt
from qtpy.QtWidgets import QFileDialog, QMainWindow, QWidget

from PartSeg.common_gui import select_multiple_files
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog, LoadProperty, PLoadDialog
from PartSeg.common_gui.custom_save_dialog import CustomSaveDialog, FormDialog, PSaveDialog
from PartSeg.common_gui.equal_column_layout import EqualColumnLayout
from PartSeg.common_gui.main_window import OPEN_DIRECTORY, OPEN_FILE, OPEN_FILE_FILTER, BaseMainWindow
from PartSeg.common_gui.multiple_file_widget import LoadRecentFiles, MultipleFileWidget, MultipleLoadDialog
from PartSeg.common_gui.qt_modal import QtPopup
from PartSeg.common_gui.searchable_combo_box import SearchComboBox
from PartSeg.common_gui.universal_gui_part import EnumComboBox
from PartSegCore.algorithm_describe_base import AlgorithmProperty
from PartSegCore.analysis.calculation_plan import MaskSuffix
from PartSegCore.analysis.load_functions import LoadProject, LoadStackImage, load_dict
from PartSegCore.analysis.save_functions import SaveAsTiff, SaveProject, save_dict
from PartSegImage import Image, ImageWriter

pyside_skip = pytest.mark.skipif(qtpy.API_NAME == "PySide2" and platform.system() == "Linux", reason="PySide2 problem")
IS_MACOS = sys.platform == "darwin"


class Enum1(Enum):
    test1 = 1
    test2 = 2
    test3 = 3


class Enum2(Enum):
    test1 = 1
    test2 = 2
    test3 = 3
    test4 = 4

    def __str__(self):
        return self.name


@pytest.mark.filterwarnings("ignore:EnumComboBox is deprecated")
class TestEnumComboBox:
    def test_enum1(self, qtbot):
        widget = EnumComboBox(Enum1)
        qtbot.addWidget(widget)
        assert widget.count() == 3
        assert widget.currentText() == "Enum1.test1"
        with qtbot.waitSignal(widget.current_choose):
            widget.set_value(Enum1.test2)

    def test_enum2(self, qtbot):
        widget = EnumComboBox(Enum2)
        qtbot.addWidget(widget)
        assert widget.count() == 4
        assert widget.currentText() == "test1"
        with qtbot.waitSignal(widget.current_choose):
            widget.set_value(Enum2.test2)


@pytest.fixture
def mock_accept_files(monkeypatch):
    def accept(*_):
        return True

    monkeypatch.setattr(select_multiple_files.AcceptFiles, "exec_", accept)


@pytest.fixture
def mock_warning(monkeypatch):
    warning_show = [0]

    def warning(*_):
        warning_show[0] = 1

    monkeypatch.setattr(select_multiple_files.QMessageBox, "warning", warning)
    return warning_show


@pytest.mark.usefixtures("mock_accept_files")
class TestAddFiles:
    def test_update_files_list(self, qtbot, tmp_path, part_settings):
        for i in range(20):
            with open(tmp_path / f"test_{i}.txt", "w") as f_p:
                f_p.write("test")
        widget = select_multiple_files.AddFiles(part_settings)
        qtbot.addWidget(widget)
        file_list1 = [str(tmp_path / f"test_{i}.txt") for i in range(15)]
        widget.update_files_list(file_list1[:10])
        assert len(widget.files_to_proceed) == 10
        widget.update_files_list(file_list1[5:])
        assert len(widget.files_to_proceed) == 15

    def test_find_all(self, qtbot, tmp_path, part_settings, mock_warning):
        for i in range(10):
            with open(tmp_path / f"test_{i}.txt", "w") as f_p:
                f_p.write("test")
        widget = select_multiple_files.AddFiles(part_settings)
        qtbot.addWidget(widget)
        widget.paths_input.setText(str(tmp_path / "*.txt"))
        widget.find_all()
        assert mock_warning[0] == 0
        assert len(widget.files_to_proceed) == 10
        widget.find_all()
        assert mock_warning[0] == 1

    def test_parse_drop_file_list(self, qtbot, tmp_path, part_settings, mock_warning):
        name_list = []
        full_name_list = []
        for i in range(10):
            with open(tmp_path / f"test_{i}.txt", "w") as f_p:
                f_p.write("test")
                name_list.append(f"test_{i}.txt")
                full_name_list.append(str(tmp_path / f"test_{i}.txt"))

        widget = select_multiple_files.AddFiles(part_settings)
        qtbot.addWidget(widget)
        widget.paths_input.setText(str(tmp_path / "aaa"))
        widget.parse_drop_file_list(name_list)
        assert mock_warning[0] == 1
        mock_warning[0] = 0
        widget.parse_drop_file_list(full_name_list)
        assert mock_warning[0] == 0
        assert len(widget.files_to_proceed) == 10
        widget.clean()
        assert len(widget.files_to_proceed) == 0
        widget.paths_input.setText(str(tmp_path))
        widget.parse_drop_file_list(name_list)
        assert mock_warning[0] == 0
        assert len(widget.files_to_proceed) == 10

    def test_delete_element(self, qtbot, tmp_path, part_settings):
        for i in range(10):
            with open(tmp_path / f"test_{i}.txt", "w") as f_p:
                f_p.write("test")
        widget = select_multiple_files.AddFiles(part_settings)
        qtbot.addWidget(widget)
        file_list = [str(tmp_path / f"test_{i}.txt") for i in range(10)]
        widget.update_files_list(file_list)
        assert len(widget.files_to_proceed) == 10
        widget.selected_files.setCurrentRow(2)
        widget.delete_element()
        assert len(widget.files_to_proceed) == 9

    def test_load_file(self, qtbot, tmp_path, part_settings):
        for i in range(10):
            with open(tmp_path / f"test_{i}.txt", "w") as f_p:
                f_p.write("test")
        widget = select_multiple_files.AddFiles(part_settings)
        qtbot.addWidget(widget)
        file_list = [str(tmp_path / f"test_{i}.txt") for i in range(10)]
        widget.update_files_list(file_list)
        widget.selected_files.setCurrentRow(2)

        def check_res(val):
            return val == [str(tmp_path / "test_2.txt")]

        with qtbot.waitSignal(part_settings.request_load_files, check_params_cb=check_res):
            widget._load_file()

        mapper = MaskSuffix(name="", suffix="_mask")

        def check_res2(val):
            return val == [str(tmp_path / "test_2.txt"), str(tmp_path / "test_2_mask.txt")]

        with qtbot.waitSignal(part_settings.request_load_files, check_params_cb=check_res2):
            widget._load_file_with_mask(mapper)


class _TestWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setLayout(EqualColumnLayout())


class TestEqualColumnLayout:
    def test_add(self, qtbot):
        widget = _TestWidget()
        qtbot.addWidget(widget)
        w1 = QWidget()
        w2 = QWidget()
        widget.layout().addWidget(w1)
        assert widget.layout().count() == 1
        widget.layout().addWidget(w2)
        assert widget.layout().count() == 2
        assert widget.layout().itemAt(1).widget() == w2
        assert widget.layout().itemAt(0).widget() == w1
        assert widget.layout().itemAt(2) is None

    def test_remove_item(self, qtbot):
        widget = _TestWidget()
        qtbot.addWidget(widget)
        w1 = QWidget()
        w2 = QWidget()
        widget.layout().addWidget(w1)
        widget.layout().addWidget(w2)
        assert widget.layout().count() == 2
        assert widget.layout().takeAt(0).widget() == w1
        assert widget.layout().itemAt(0).widget() == w2
        assert widget.layout().count() == 1
        assert widget.layout().takeAt(2) is None

    @pyside_skip
    def test_geometry(self, qtbot):
        widget = _TestWidget()
        qtbot.addWidget(widget)
        w1 = QWidget()
        w2 = QWidget()
        widget.layout().addWidget(w1)
        widget.layout().addWidget(w2)
        widget.show()
        widget.resize(200, 200)
        assert widget.width() == 200
        assert w1.width() == 100
        widget.hide()

    @pyside_skip
    def test_hidden_widget(self, qtbot):
        widget = _TestWidget()
        w1 = QWidget()
        w2 = QWidget()
        w3 = QWidget()
        widget.layout().addWidget(w1)
        widget.layout().addWidget(w2)
        widget.layout().addWidget(w3)
        w2.hide()
        qtbot.addWidget(widget)
        widget.show()
        widget.resize(200, 200)
        assert w1.width() == 100
        widget.hide()


class TestSearchCombBox:
    def test_create(self, qtbot):
        widget = SearchComboBox()
        qtbot.addWidget(widget)

    def test_add_item(self, qtbot):
        widget = SearchComboBox()
        qtbot.addWidget(widget)
        widget.addItem("test1")
        assert widget.count() == 1
        assert widget.itemText(0) == "test1"

    def test_add_items(self, qtbot):
        widget = SearchComboBox()
        qtbot.addWidget(widget)
        widget.addItems(["test1", "test2", "test3"])
        assert widget.count() == 3
        assert widget.itemText(0) == "test1"
        assert widget.itemText(2) == "test3"


def test_create_load_dialog(qtbot):
    dialog = CustomLoadDialog(load_dict, history=["/aaa/"])
    assert dialog.acceptMode() == CustomLoadDialog.AcceptOpen
    dialog = CustomLoadDialog(LoadProject, history=["/aaa/"])
    assert dialog.acceptMode() == CustomLoadDialog.AcceptOpen


def test_create_save_dialog(qtbot):
    dialog = CustomSaveDialog(save_dict, history=["/aaa/"])
    assert dialog.acceptMode() == CustomSaveDialog.AcceptSave
    dialog = CustomSaveDialog(SaveProject, history=["/aaa/"])
    assert not hasattr(dialog, "stack_widget")
    dialog = CustomSaveDialog(save_dict, system_widget=False)
    assert hasattr(dialog, "stack_widget")


def test_p_save_dialog(part_settings, tmp_path, qtbot, monkeypatch):
    def selected_files(self):
        return [str(tmp_path / "test.tif")]

    monkeypatch.setattr(QFileDialog, "selectedFiles", selected_files)

    assert part_settings.get_path_history() == [str(Path.home())]

    dialog = PSaveDialog(save_dict, settings=part_settings, path="io.test")
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == Path.home()
    assert Path(part_settings.get("io.test")) == Path.home()
    dialog = PSaveDialog(save_dict, settings=part_settings, path="io.test2", default_directory=str(tmp_path))
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == tmp_path
    assert Path(part_settings.get("io.test2")) == tmp_path
    part_settings.set("io.test3", str(tmp_path))
    dialog = PSaveDialog(save_dict, settings=part_settings, path="io.test3")
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == tmp_path
    assert Path(part_settings.get("io.test3")) == tmp_path

    monkeypatch.setattr(QFileDialog, "result", lambda x: QFileDialog.Rejected)
    part_settings.set("io.filter_save", SaveAsTiff.get_name())
    assert part_settings.get_path_history() == [str(Path.home())]
    dialog.show()
    dialog.accept()
    assert part_settings.get_path_history() == [str(Path.home())]

    monkeypatch.setattr(QFileDialog, "result", lambda x: QFileDialog.Accepted)
    dialog = PSaveDialog(save_dict, settings=part_settings, path="io.test4", filter_path="io.filter_save")
    qtbot.addWidget(dialog)
    assert SaveAsTiff.get_name() in dialog.nameFilters()
    dialog.show()
    dialog.selectFile(str(tmp_path / "test.tif"))
    dialog.accept()
    assert dialog.selectedNameFilter() == SaveAsTiff.get_name()
    assert [Path(x) for x in part_settings.get_path_history()] == [tmp_path, Path.home()]


def test_form_dialog(qtbot):
    fields = [
        AlgorithmProperty("aaa", "Aaa", 1.0),
        AlgorithmProperty("bbb", "Bbb", False),
    ]
    form = FormDialog(fields, values={"aaa": 2.0})
    assert form.get_values() == {"aaa": 2.0, "bbb": False}
    form.set_values({"aaa": 5.0, "bbb": True})
    assert form.get_values() == {"aaa": 5.0, "bbb": True}


def test_p_load_dialog(part_settings, tmp_path, qtbot, monkeypatch):
    dialog = PLoadDialog(load_dict, settings=part_settings, path="io.load_test")
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == Path.home()
    assert Path(part_settings.get("io.load_test")) == Path.home()
    dialog = PLoadDialog(load_dict, settings=part_settings, path="io.load_test2", default_directory=str(tmp_path))
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == tmp_path
    assert Path(part_settings.get("io.load_test2")) == tmp_path
    part_settings.set("io.load_test3", str(tmp_path))
    dialog = PLoadDialog(load_dict, settings=part_settings, path="io.load_test3")
    qtbot.addWidget(dialog)
    assert Path(dialog.directory().path()) == tmp_path
    assert Path(part_settings.get("io.load_test3")) == tmp_path

    monkeypatch.setattr(QFileDialog, "result", lambda x: QFileDialog.Rejected)
    part_settings.set("io.filter_load", LoadStackImage.get_name())
    assert part_settings.get_path_history() == [str(Path.home())]
    dialog.show()
    dialog.accept()
    assert part_settings.get_path_history() == [str(Path.home())]

    with (tmp_path / "test.tif").open("w") as f:
        f.write("eeeeeee")

    monkeypatch.setattr(QFileDialog, "result", lambda x: QFileDialog.Accepted)
    dialog = PLoadDialog(load_dict, settings=part_settings, path="io.load_test4", filter_path="io.filter_load")
    qtbot.addWidget(dialog)
    assert LoadStackImage.get_name() in dialog.nameFilters()
    dialog.show()
    dialog.selectFile(str(tmp_path / "test.tif"))
    if IS_MACOS:
        monkeypatch.setattr(dialog, "selectedFiles", lambda: [str(tmp_path / "test.tif")])
    dialog.accept()
    assert dialog.selectedNameFilter() == LoadStackImage.get_name()
    assert [Path(x) for x in part_settings.get_path_history()] == [tmp_path, Path.home()]


def test_str_filter(part_settings, tmp_path, qtbot, monkeypatch):
    tiff_text = "Test (*.tiff)"
    monkeypatch.setattr(QFileDialog, "result", lambda x: QFileDialog.Accepted)
    monkeypatch.setattr(QFileDialog, "selectedFiles", lambda x: [str(tmp_path / "test.tif")])
    dialog = PSaveDialog(tiff_text, settings=part_settings, path="io.save_test")
    qtbot.addWidget(dialog)
    assert tiff_text in dialog.nameFilters()
    dialog.show()
    dialog.selectFile(str(tmp_path / "test.tif"))
    dialog.accept()
    assert dialog.selectedNameFilter() == tiff_text
    assert [Path(x) for x in part_settings.get_path_history()] == [tmp_path, Path.home()]

    with (tmp_path / "test2.tif").open("w") as f:
        f.write("eeeeeee")

    dialog = PLoadDialog(tiff_text, settings=part_settings, path="io.load_test2")
    qtbot.addWidget(dialog)
    assert tiff_text in dialog.nameFilters()
    dialog.show()
    dialog.selectFile(str(tmp_path / "test2.tif"))
    if IS_MACOS:
        monkeypatch.setattr(dialog, "selectedFiles", lambda: [str(tmp_path / "test2.tif")])
    dialog.accept()
    assert dialog.selectedNameFilter() == tiff_text
    assert [Path(x) for x in part_settings.get_path_history()] == [tmp_path, Path.home()]


def test_recent_files(part_settings, qtbot):
    dial = LoadRecentFiles(part_settings)
    qtbot.add_widget(dial)
    assert dial.file_list.count() == 0
    size = dial.size()
    new_size = size.width() + 50, size.width() + 50
    dial.resize(*new_size)
    dial.accept()
    assert part_settings.get_from_profile("multiple_files_dialog_size") == new_size
    part_settings.add_last_files_multiple(["aaa.txt"], "method")
    part_settings.add_last_files_multiple(["bbb.txt"], "method")
    part_settings.add_last_files(["bbb.txt"], "method")
    part_settings.add_last_files(["ccc.txt"], "method")
    dial = LoadRecentFiles(part_settings)
    qtbot.add_widget(dial)
    assert dial.file_list.count() == 3
    assert dial.size() == QSize(*new_size)
    dial.file_list.selectAll()
    assert dial.get_files() == [(["bbb.txt"], "method"), (["aaa.txt"], "method"), (["ccc.txt"], "method")]


class TestMultipleFileWidget:
    def test_create(self, part_settings, qtbot):
        widget = MultipleFileWidget(part_settings, {})
        qtbot.add_widget(widget)

    @staticmethod
    def check_load_files(parameter, custom_name):
        return not custom_name and os.path.basename(parameter.file_path) == "img_4.tif"

    @pytest.mark.enablethread
    @pytest.mark.enabledialog
    def test_load_recent(self, part_settings, qtbot, monkeypatch, tmp_path):
        widget = MultipleFileWidget(part_settings, {LoadStackImage.get_name(): LoadStackImage})
        qtbot.add_widget(widget)
        for i in range(5):
            ImageWriter.save(
                Image(np.random.random((10, 10)), image_spacing=(1, 1), axes_order="XY"), tmp_path / f"img_{i}.tif"
            )
        file_list = [
            [
                [
                    tmp_path / f"img_{i}.tif",
                ],
                LoadStackImage.get_name(),
            ]
            for i in range(5)
        ]
        with qtbot.waitSignal(widget._add_state, check_params_cb=self.check_load_files):
            widget.load_recent_fun(file_list, lambda x, y: True, lambda x: True)
        assert part_settings.get_last_files_multiple() == file_list
        assert widget.file_view.topLevelItemCount() == 5
        widget.file_view.clear()
        widget.state_dict.clear()
        widget.file_list.clear()
        monkeypatch.setattr(LoadRecentFiles, "exec_", lambda x: True)
        monkeypatch.setattr(LoadRecentFiles, "get_files", lambda x: file_list)
        with qtbot.waitSignal(widget._add_state, check_params_cb=self.check_load_files):
            widget.load_recent()
        assert part_settings.get_last_files_multiple() == file_list
        assert widget.file_view.topLevelItemCount() == 5

    @pytest.mark.enablethread
    @pytest.mark.enabledialog
    def test_load_files(self, part_settings, qtbot, monkeypatch, tmp_path):
        widget = MultipleFileWidget(part_settings, {LoadStackImage.get_name(): LoadStackImage})
        qtbot.add_widget(widget)
        for i in range(5):
            ImageWriter.save(
                Image(np.random.random((10, 10)), image_spacing=(1, 1), axes_order="XY"), tmp_path / f"img_{i}.tif"
            )
        file_list = [[[str(tmp_path / f"img_{i}.tif")], LoadStackImage.get_name()] for i in range(5)]
        load_property = LoadProperty(
            [str(tmp_path / f"img_{i}.tif") for i in range(5)], LoadStackImage.get_name(), LoadStackImage
        )
        with qtbot.waitSignal(widget._add_state, check_params_cb=self.check_load_files):
            widget.execute_load_files(load_property, lambda x, y: True, lambda x: True)
        assert widget.file_view.topLevelItemCount() == 5
        assert part_settings.get_last_files_multiple() == file_list
        widget.file_view.clear()
        widget.state_dict.clear()
        widget.file_list.clear()
        monkeypatch.setattr(MultipleLoadDialog, "exec_", lambda x: True)
        monkeypatch.setattr(MultipleLoadDialog, "get_result", lambda x: load_property)
        with qtbot.waitSignal(widget._add_state, check_params_cb=self.check_load_files):
            widget.load_files()
        assert widget.file_view.topLevelItemCount() == 5
        assert part_settings.get_last_files_multiple() == file_list
        part_settings.dump()
        part_settings.load()
        assert part_settings.get_last_files_multiple() == file_list


class TestBaseMainWindow:
    def test_create(self, tmp_path, qtbot):
        window = BaseMainWindow(config_folder=tmp_path)
        qtbot.add_widget(window)

    @pytest.mark.enablethread
    @pytest.mark.enabledialog
    def test_recent(self, tmp_path, qtbot, monkeypatch):
        load_mock = MagicMock()
        load_mock.load = MagicMock(return_value=1)
        load_mock.get_name = MagicMock(return_value="test")
        window = BaseMainWindow(config_folder=tmp_path, load_dict={"test": load_mock})
        qtbot.add_widget(window)
        assert window.recent_file_menu.isEmpty()
        window.settings.add_last_files([tmp_path / "test.txt"], "test")
        actions = window.recent_file_menu.actions()
        assert len(actions) == 1
        assert actions[0].data() == ([tmp_path / "test.txt"], "test")
        monkeypatch.setattr(window, "sender", lambda: actions[0])
        main_menu = MagicMock()
        add_last_files = MagicMock()
        monkeypatch.setattr(window, "main_menu", main_menu, raising=False)
        monkeypatch.setattr(window.settings, "add_last_files", add_last_files)
        window._load_recent()
        window.settings.add_last_files.assert_called_once_with([tmp_path / "test.txt"], "test")
        main_menu.set_data.assert_called_with(1)
        assert window.settings.get(OPEN_DIRECTORY) == str(tmp_path)
        assert str(window.settings.get(OPEN_FILE)) == str(tmp_path / "test.txt")
        assert window.settings.get(OPEN_FILE_FILTER) == "test"


class TestQtPopup:
    def test_show_above(self, qtbot):
        popup = QtPopup(None)
        qtbot.addWidget(popup)
        popup.show_above_mouse()
        popup.close()

    def test_show_right(self, qtbot):
        popup = QtPopup(None)
        qtbot.addWidget(popup)
        popup.show_right_of_mouse()
        popup.close()

    def test_move_to_error_no_parent(self, qtbot):
        popup = QtPopup(None)
        qtbot.add_widget(popup)
        with pytest.raises(ValueError):
            popup.move_to()

    @pytest.mark.parametrize("pos", ["top", "bottom", "left", "right"])
    def test_move_to(self, pos, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        popup.move_to(pos)

    def test_move_to_error_wrong_params(self, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        with pytest.raises(ValueError):
            popup.move_to("dummy_text")

        with pytest.raises(ValueError):
            popup.move_to({})

    @pytest.mark.parametrize("pos", [[10, 10, 10, 10], (15, 10, 10, 10)])
    def test_move_to_cords(self, pos, qtbot):
        window = QMainWindow()
        qtbot.addWidget(window)
        widget = QWidget()
        window.setCentralWidget(widget)
        popup = QtPopup(widget)
        popup.move_to(pos)

    def test_click(self, qtbot, monkeypatch):
        popup = QtPopup(None)
        monkeypatch.setattr(popup, "close", MagicMock())
        qtbot.addWidget(popup)
        qtbot.keyClick(popup, Qt.Key_8)
        popup.close.assert_not_called()
        qtbot.keyClick(popup, Qt.Key_Return)
        popup.close.assert_called_once()
