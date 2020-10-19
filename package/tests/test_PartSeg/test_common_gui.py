import platform
from enum import Enum

import pytest
import qtpy
from qtpy.QtWidgets import QWidget

from PartSeg.common_gui import select_multiple_files
from PartSeg.common_gui.equal_column_layout import EqualColumnLayout
from PartSeg.common_gui.searchable_combo_box import SearchCombBox
from PartSeg.common_gui.universal_gui_part import EnumComboBox
from PartSegCore.analysis.calculation_plan import MaskSuffix


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

    @pytest.mark.skipif(qtpy.API_NAME == "PySide2" and platform.system() == "Linux", reason="PySide2 problem")
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

    @pytest.mark.skipif(qtpy.API_NAME == "PySide2" and platform.system() == "Linux", reason="PySide2 problem")
    def test_hidden_widget(self, qtbot):
        widget = _TestWidget()
        qtbot.addWidget(widget)
        w1 = QWidget()
        w2 = QWidget()
        w3 = QWidget()
        widget.layout().addWidget(w1)
        widget.layout().addWidget(w2)
        widget.layout().addWidget(w3)
        w2.hide()
        widget.show()
        widget.resize(200, 200)
        assert w1.width() == 100
        widget.hide()


class TestSearchCombBox:
    def test_create(self, qtbot):
        widget = SearchCombBox()
        qtbot.addWidget(widget)

    def test_add_item(self, qtbot):
        widget = SearchCombBox()
        qtbot.addWidget(widget)
        widget.addItem("test1")
        assert widget.count() == 1
        assert widget.itemText(0) == "test1"

    def test_add_items(self, qtbot):
        widget = SearchCombBox()
        qtbot.addWidget(widget)
        widget.addItems(["test1", "test2", "test3"])
        assert widget.count() == 3
        assert widget.itemText(0) == "test1"
        assert widget.itemText(2) == "test3"
