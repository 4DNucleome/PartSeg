# pylint: disable=R0201
import datetime
import os
import platform
import subprocess  # nosec
import sys
import typing
from enum import Enum
from functools import partial
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import qtpy
from magicgui.widgets import Widget
from nme import register_class
from pydantic import Field
from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)
from superqt import QEnumComboBox

from PartSeg.common_gui import exception_hooks, select_multiple_files
from PartSeg.common_gui.about_dialog import AboutDialog
from PartSeg.common_gui.advanced_tabs import (
    RENDERING_LIST,
    RENDERING_MODE_NAME_STR,
    SEARCH_ZOOM_FACTOR_STR,
    AdvancedWindow,
    Appearance,
)
from PartSeg.common_gui.algorithms_description import (
    AlgorithmChoose,
    AlgorithmChooseBase,
    BaseAlgorithmSettingsWidget,
    FieldsList,
    FormWidget,
    InteractiveAlgorithmSettingsWidget,
    ListInput,
    ProfileSelect,
    QtAlgorithmProperty,
    SubAlgorithmWidget,
)
from PartSeg.common_gui.collapse_checkbox import CollapseCheckbox
from PartSeg.common_gui.custom_load_dialog import (
    CustomLoadDialog,
    IOMethodMock,
    LoadProperty,
    LoadRegisterFileDialog,
    PLoadDialog,
)
from PartSeg.common_gui.custom_save_dialog import CustomSaveDialog, FormDialog, PSaveDialog
from PartSeg.common_gui.equal_column_layout import EqualColumnLayout
from PartSeg.common_gui.error_report import DataImportErrorDialog
from PartSeg.common_gui.image_adjustment import ImageAdjustmentDialog, ImageAdjustTuple
from PartSeg.common_gui.main_window import OPEN_DIRECTORY, OPEN_FILE, OPEN_FILE_FILTER, BaseMainWindow
from PartSeg.common_gui.mask_widget import MaskDialogBase, MaskWidget
from PartSeg.common_gui.multiple_file_widget import (
    LoadRecentFiles,
    MultipleFilesTreeWidget,
    MultipleFileWidget,
    MultipleLoadDialog,
)
from PartSeg.common_gui.qt_modal import QtPopup
from PartSeg.common_gui.searchable_combo_box import SearchComboBox
from PartSeg.common_gui.show_directory_dialog import DirectoryDialog
from PartSeg.common_gui.universal_gui_part import (
    ChannelComboBox,
    CustomDoubleSpinBox,
    CustomSpinBox,
    EnumComboBox,
    InfoLabel,
    Spacing,
)
from PartSegCore import Units, state_store
from PartSegCore.algorithm_describe_base import (
    AlgorithmDescribeBase,
    AlgorithmProperty,
    AlgorithmSelection,
    Register,
    ROIExtractionProfile,
    base_model_to_algorithm_property,
)
from PartSegCore.analysis import AnalysisAlgorithmSelection
from PartSegCore.analysis.calculation_plan import MaskSuffix
from PartSegCore.analysis.load_functions import LoadProject, LoadStackImage, load_dict
from PartSegCore.analysis.save_functions import SaveAsTiff, SaveProject, save_dict
from PartSegCore.image_operations import RadiusType
from PartSegCore.io_utils import LoadPlanExcel, LoadPlanJson, SaveBase
from PartSegCore.mask_create import MaskProperty
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegCore.segmentation.restartable_segmentation_algorithms import BorderRim, LowerThresholdAlgorithm
from PartSegCore.utils import BaseModel
from PartSegImage import Channel, Image, ImageWriter
from PartSegImage.image_reader import INCOMPATIBLE_IMAGE_MASK

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
        return f"{self.name} eee"


@pytest.mark.filterwarnings("ignore:EnumComboBox is deprecated")
class TestEnumComboBox:
    def test_enum1(self, qtbot):
        widget = EnumComboBox(Enum1)
        qtbot.addWidget(widget)
        assert widget.count() == 3
        assert widget.currentText() == "test1"
        with qtbot.waitSignal(widget.current_choose):
            widget.set_value(Enum1.test2)

    def test_enum2(self, qtbot):
        widget = EnumComboBox(Enum2)
        qtbot.addWidget(widget)
        assert widget.count() == 4
        assert widget.currentText() == "test1 eee"
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
    result = dialog.get_result()
    assert result.load_class is LoadProject
    assert result.selected_filter == LoadProject.get_name_with_suffix()
    assert result.load_location == []


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

    @pytest.mark.parametrize("pos", ["top", "bottom", "left", "right", (10, 10, 10, 10), (15, 10, 10, 10)])
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

    def test_click(self, qtbot, monkeypatch):
        popup = QtPopup(None)
        monkeypatch.setattr(popup, "close", MagicMock())
        qtbot.addWidget(popup)
        qtbot.keyClick(popup, Qt.Key_8)
        popup.close.assert_not_called()
        qtbot.keyClick(popup, Qt.Key_Return)
        popup.close.assert_called_once()


@pytest.mark.parametrize("function_name", SaveBase.need_functions)
def test_IOMethodMock(function_name):
    Register.check_function(IOMethodMock("test"), function_name, True)
    getattr(IOMethodMock("test"), function_name)()


def test_about_dialog_create(qtbot):
    dialog = AboutDialog()
    qtbot.addWidget(dialog)


class TestAdvancedWindow:
    def test_create_no_develop(self, monkeypatch, qtbot, base_settings):
        monkeypatch.setattr(state_store, "develop", False)
        wind = AdvancedWindow(base_settings, [])
        qtbot.add_widget(wind)
        assert wind.indexOf(wind.develop) == -1

    def test_create_with_develop(self, monkeypatch, qtbot, base_settings):
        monkeypatch.setattr(state_store, "develop", True)
        wind = AdvancedWindow(base_settings, [])
        qtbot.add_widget(wind)
        assert wind.indexOf(wind.develop) != -1


class TestAppearance:
    def test_change_zoom_factor(self, qtbot, base_settings):
        base_settings.set_in_profile(SEARCH_ZOOM_FACTOR_STR, 1.5)
        app = Appearance(base_settings)
        qtbot.add_widget(app)
        assert app.zoom_factor_spin_box.value() == 1.5
        base_settings.set_in_profile(SEARCH_ZOOM_FACTOR_STR, 2)
        app = Appearance(base_settings)
        qtbot.add_widget(app)
        assert app.zoom_factor_spin_box.value() == 2

        with qtbot.wait_signal(app.zoom_factor_spin_box.valueChanged):
            app.zoom_factor_spin_box.setValue(1)
        assert base_settings.get_from_profile(SEARCH_ZOOM_FACTOR_STR) == 1

        base_settings.set_in_profile(SEARCH_ZOOM_FACTOR_STR, 2)
        assert app.zoom_factor_spin_box.value() == 2

    @pytest.mark.skipif(len(RENDERING_LIST) <= 1, reason="no rendering to switch")
    def test_rendering_select(self, qtbot, base_settings):
        base_settings.set_in_profile(RENDERING_MODE_NAME_STR, RENDERING_LIST[0])
        app = Appearance(base_settings)
        qtbot.add_widget(app)
        assert app.labels_render_cmb.currentText() == RENDERING_LIST[0]
        with qtbot.wait_signal(app.labels_render_cmb.currentIndexChanged):
            app.labels_render_cmb.setCurrentIndex(1)
        assert app.labels_render_cmb.currentText() == RENDERING_LIST[1]
        assert base_settings.get_from_profile(RENDERING_MODE_NAME_STR) == RENDERING_LIST[1]
        base_settings.set_in_profile(RENDERING_MODE_NAME_STR, RENDERING_LIST[0])
        assert app.labels_render_cmb.currentText() == RENDERING_LIST[0]

    def test_theme_select(self, qtbot, base_settings, monkeypatch):
        monkeypatch.setattr(base_settings, "theme_list", lambda: ["aaa", "bbb", "ccc"])
        monkeypatch.setattr(base_settings, "napari_settings", MagicMock())
        monkeypatch.setattr(base_settings, "theme_name", "bbb")
        app = Appearance(base_settings)
        qtbot.add_widget(app)
        assert app.layout_list.currentText() == "bbb"
        app.layout_list.setCurrentIndex(0)
        assert app.layout_list.currentText() == "aaa"
        assert base_settings.theme_name == "aaa"


class TestFormWidget:
    def test_create(self, qtbot):
        form = FormWidget([])
        qtbot.add_widget(form)
        assert not form.has_elements()
        assert form.get_values() == {}

    def test_single_field_widget(self, qtbot):
        form = FormWidget([AlgorithmProperty("test", "Test", 1)])
        qtbot.add_widget(form)
        assert form.has_elements()
        assert form.get_values() == {"test": 1}

    def test_single_field_widget_start_values(self, qtbot):
        form = FormWidget([AlgorithmProperty("test", "Test", 1)], start_values={"test": 2})
        qtbot.add_widget(form)
        assert form.has_elements()
        assert form.get_values() == {"test": 2}
        form.set_values({"test": 5})
        assert form.get_values() == {"test": 5}

    def test_single_field_widget_wrong_start_values(self, qtbot):
        form = FormWidget([AlgorithmProperty("test", "Test", 1)], start_values={"test": "aaa"})
        qtbot.add_widget(form)
        assert form.has_elements()
        assert form.get_values() == {"test": 1}

    def test_base_model_simple_create(self, qtbot):
        class Fields(BaseModel):
            test: int = 5

        form = FormWidget(Fields)
        qtbot.add_widget(form)
        assert form.has_elements()
        assert isinstance(form.get_values(), Fields)
        assert form.get_values() == Fields(test=5)
        form.set_values(Fields(test=8))
        assert form.get_values() == Fields(test=8)

    def test_base_model_nested_create(self, qtbot):
        class SubFields(BaseModel):
            field1: int = 0
            field2: float = 0

        class Fields(BaseModel):
            test1: SubFields = SubFields(field1=5, field2=7)
            test2: SubFields = SubFields(field1=15, field2=41)

        form = FormWidget(Fields)
        qtbot.add_widget(form)
        assert form.has_elements()
        assert isinstance(form.get_values(), Fields)
        assert form.get_values() == Fields(test1=SubFields(field1=5, field2=7), test2=SubFields(field1=15, field2=41))

    def test_base_model_register_create(self, qtbot):
        class SampleSelection(AlgorithmSelection):
            pass

        class SampleClass1(AlgorithmDescribeBase):
            @classmethod
            def get_name(cls) -> str:
                return "1"

            @classmethod
            def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
                return [AlgorithmProperty("field", "Field", 1)]

        class SampleClass2(AlgorithmDescribeBase):
            @classmethod
            def get_name(cls) -> str:
                return "2"

            @classmethod
            def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
                return [AlgorithmProperty("field_", "Field", 2)]

        SampleSelection.register(SampleClass1)
        SampleSelection.register(SampleClass2)

        class SampleModel(BaseModel):
            field1: int = Field(10, le=100, ge=0, title="Field 1")
            check_selection: SampleSelection = Field(SampleSelection(name="1", values={}), title="Class selection")

        form = FormWidget(SampleModel)
        qtbot.add_widget(form)
        assert form.has_elements()
        assert isinstance(form.get_values(), SampleModel)
        assert form.get_values() == SampleModel(
            field1=10, check_selection=SampleSelection(name="1", values={"field": 1})
        )

    def test_base_model_register_nested_create(self, qtbot, clean_register):
        class SampleSelection(AlgorithmSelection):
            pass

        @register_class
        class SubModel1(BaseModel):
            field1: int = 3

        @register_class
        class SubModel2(BaseModel):
            field2: int = 5

        class SampleClass1(AlgorithmDescribeBase):
            __argument_class__ = SubModel1

            @classmethod
            def get_name(cls) -> str:
                return "1"

        class SampleClass2(AlgorithmDescribeBase):
            __argument_class__ = SubModel2

            @classmethod
            def get_name(cls) -> str:
                return "2"

        SampleSelection.register(SampleClass1)
        SampleSelection.register(SampleClass2)

        @register_class
        class SampleModel(BaseModel):
            field1: int = Field(10, le=100, ge=0, title="Field 1")
            check_selection: SampleSelection = Field(SampleSelection(name="1", values={}), title="Class selection")

        form = FormWidget(SampleModel)
        qtbot.add_widget(form)
        assert form.has_elements()
        assert isinstance(form.get_values(), SampleModel)
        assert isinstance(form.get_values().check_selection.values, SubModel1)
        assert form.get_values() == SampleModel(
            field1=10, check_selection=SampleSelection(name="1", values=SubModel1(field1=3))
        )

    def test_image_changed(self, qtbot, clean_register, image):
        class SampleSelection(AlgorithmSelection):
            pass

        @register_class
        class SubModel1(BaseModel):
            field1: Channel = 1

        @register_class
        class SubModel2(BaseModel):
            field2: Channel = 0

        class SampleClass1(AlgorithmDescribeBase):
            __argument_class__ = SubModel1

            @classmethod
            def get_name(cls) -> str:
                return "1"

        class SampleClass2(AlgorithmDescribeBase):
            __argument_class__ = SubModel2

            @classmethod
            def get_name(cls) -> str:
                return "2"

        SampleSelection.register(SampleClass1)
        SampleSelection.register(SampleClass2)

        @register_class
        class SampleModel(BaseModel):
            field1: int = Field(10, le=100, ge=0, title="Field 1")
            channel: Channel = Field(1, title="Channel")
            check_selection: SampleSelection = Field(SampleSelection(name="1", values={}), title="Class selection")

        form = FormWidget(SampleModel)
        qtbot.add_widget(form)
        assert len(form.channels_chose) == 2
        assert isinstance(form.channels_chose[0], ChannelComboBox)
        assert isinstance(form.channels_chose[1], SubAlgorithmWidget)
        assert form.channels_chose[0].count() == 10
        form.image_changed(None)
        assert form.channels_chose[0].count() == 10
        form.image_changed(image)
        assert form.channels_chose[0].count() == 2

    def test_recursive_get_values(self, qtbot, clean_register):
        class SampleSelection(AlgorithmSelection):
            pass

        @register_class
        class SubModel1(BaseModel):
            field1: int = 3

        @register_class
        class SubModel2(BaseModel):
            field2: int = 5

        class SampleClass1(AlgorithmDescribeBase):
            __argument_class__ = SubModel1

            @classmethod
            def get_name(cls) -> str:
                return "1"

        class SampleClass2(AlgorithmDescribeBase):
            __argument_class__ = SubModel2

            @classmethod
            def get_name(cls) -> str:
                return "2"

        SampleSelection.register(SampleClass1)
        SampleSelection.register(SampleClass2)

        @register_class
        class SampleModel(BaseModel):
            field1: int = Field(10, le=100, ge=0, title="Field 1")
            check_selection: SampleSelection = Field(SampleSelection(name="1", values={}), title="Class selection")

        form = FormWidget(SampleModel)
        qtbot.add_widget(form)
        assert form.recursive_get_values() == {"field1": 10, "check_selection": {"1": {"field1": 3}}}
        assert form.widgets_dict["check_selection"].get_field().choose.currentText() == "1"
        assert form.widgets_dict["check_selection"].get_field().choose.currentIndex() == 0
        assert form.widgets_dict["check_selection"].get_field().choose.count() == 2
        form.widgets_dict["check_selection"].get_field().choose.setCurrentIndex(1)
        assert form.widgets_dict["check_selection"].get_field().choose.currentText() == "2"
        assert form.recursive_get_values() == {
            "field1": 10,
            "check_selection": {"1": {"field1": 3}, "2": {"field2": 5}},
        }


class TestFieldsList:
    def test_simple(self):
        li = FieldsList([])
        assert li.get_value() == {}

    def test_setting_values(self, qtbot):
        class Fields(BaseModel):
            field1: int = 1
            field2: float = 3

        ap_li = [QtAlgorithmProperty.from_algorithm_property(x) for x in base_model_to_algorithm_property(Fields)]
        for el in ap_li:
            qtbot.add_widget(el.get_field())

        li = FieldsList(ap_li)
        assert li.get_value() == {"field1": 1, "field2": 3}

        li.set_value(Fields(field1=2, field2=1))
        assert li.get_value() == {"field1": 2, "field2": 1}

        li.set_value({"field1": 5, "field2": 8})
        assert li.get_value() == {"field1": 5, "field2": 8}

    def test_signal(self, qtbot):
        ap = QtAlgorithmProperty(name="test", user_name="Test", default_value=1)
        qtbot.add_widget(ap.get_field())
        li = FieldsList([ap])
        with qtbot.wait_signal(li.changed):
            ap.set_value(3)
        assert li.get_value() == {"test": 3}


class EnumQtAl(Enum):
    test1 = 1
    test2 = 2


class ModelQtAl(BaseModel):
    field1 = 1
    field2 = 2.0


class TestQtAlgorithmProperty:
    def test_from_algorithm_property(self, qtbot):
        res = QtAlgorithmProperty.from_algorithm_property("aaaa")
        assert isinstance(res, QLabel)

        ap = AlgorithmProperty(name="test", user_name="Test", default_value=1)
        res = QtAlgorithmProperty.from_algorithm_property(ap)
        assert isinstance(res, QtAlgorithmProperty)
        qtbot.add_widget(res.get_field())

        with pytest.raises(ValueError):
            QtAlgorithmProperty.from_algorithm_property(1)

    @pytest.mark.parametrize(
        "data_type,default_value,expected_type,next_value",
        [
            (Channel, Channel(1), ChannelComboBox, Channel(2)),
            (bool, True, QCheckBox, False),
            (int, 1, CustomSpinBox, 2),
            (float, 1, CustomDoubleSpinBox, 3),
            (EnumQtAl, EnumQtAl.test1, QEnumComboBox, EnumQtAl.test2),
            (str, "a", QLineEdit, "b"),
            (ModelQtAl, ModelQtAl(), FieldsList, ModelQtAl(field1=3, field2=4.5)),
            (datetime.date, datetime.date(2022, 3, 24), Widget, datetime.date(2022, 3, 25)),
        ],
    )
    def test_types(self, qtbot, data_type, default_value, expected_type, next_value):
        ap = AlgorithmProperty(name="test", user_name="Test", default_value=default_value, value_type=data_type)
        res = QtAlgorithmProperty.from_algorithm_property(ap)
        if isinstance(res.get_field(), FieldsList):
            for el in res.get_field().field_list:
                qtbot.add_widget(el.get_field())
        elif isinstance(res.get_field(), QWidget):
            qtbot.add_widget(res.get_field())
        assert isinstance(res.get_field(), expected_type)
        assert res.get_value() == default_value
        with qtbot.wait_signal(res.change_fun):
            res.set_value(next_value)
        assert res.get_value() == next_value

    def test_list_type(self, qtbot):
        ap = AlgorithmProperty(
            name="test", user_name="Test", default_value="aaa", value_type=list, possible_values=["aaa", "bbb"]
        )
        res = QtAlgorithmProperty.from_algorithm_property(ap)
        qtbot.add_widget(res.get_field())
        assert isinstance(res.get_field(), QComboBox)
        assert res.get_value() == "aaa"

        with qtbot.wait_signal(res.change_fun):
            res.set_value("bbb")
        assert res.get_value() == "bbb"

    def test_numeric_type_default_value_error(self):
        ap = AlgorithmProperty(name="test", user_name="Test", default_value="a", value_type=int)
        with pytest.raises(ValueError):
            QtAlgorithmProperty.from_algorithm_property(ap)
        ap.default_value = 1.0
        with pytest.raises(ValueError):
            QtAlgorithmProperty.from_algorithm_property(ap)
        ap = AlgorithmProperty(name="test", user_name="Test", default_value="a", value_type=float)
        with pytest.raises(ValueError):
            QtAlgorithmProperty.from_algorithm_property(ap)

    def test_per_dimension(self, qtbot):
        ap = AlgorithmProperty(name="test", user_name="Test", default_value=1, per_dimension=True)
        res = QtAlgorithmProperty.from_algorithm_property(ap)
        qtbot.add_widget(res.get_field())
        assert isinstance(res.get_field(), ListInput)
        assert res.get_value() == [1, 1, 1]
        with qtbot.wait_signal(res.change_fun):
            res.set_value([2, 4, 6])
        assert res.get_value() == [2, 4, 6]

        with qtbot.wait_signal(res.change_fun):
            res.set_value(1)

        assert res.get_value() == [1, 1, 1]


SAMPLE_FILTER = "Sample text (*.txt)"
HEADER = "Header"


class TestLoadRegisterFileDialog:
    def test_str_register(self):
        dialog = LoadRegisterFileDialog(SAMPLE_FILTER, HEADER)
        assert len(dialog.io_register) == 1
        assert isinstance(dialog.io_register[SAMPLE_FILTER], IOMethodMock)

    def test_single_entry(self):
        dialog = LoadRegisterFileDialog(LoadPlanJson, HEADER)
        assert len(dialog.io_register) == 1
        assert issubclass(dialog.io_register[LoadPlanJson.get_name()], LoadPlanJson)

    def test_list_register(self):
        dialog = LoadRegisterFileDialog([LoadPlanJson, LoadPlanExcel], HEADER)
        assert len(dialog.io_register) == 2
        assert issubclass(dialog.io_register[LoadPlanJson.get_name()], LoadPlanJson)
        assert issubclass(dialog.io_register[LoadPlanExcel.get_name()], LoadPlanExcel)

    def test_dict_register(self):
        dialog = LoadRegisterFileDialog(
            {LoadPlanJson.get_name(): LoadPlanJson, LoadPlanExcel.get_name(): LoadPlanExcel}, HEADER
        )
        assert len(dialog.io_register) == 2
        assert issubclass(dialog.io_register[LoadPlanJson.get_name()], LoadPlanJson)
        assert issubclass(dialog.io_register[LoadPlanExcel.get_name()], LoadPlanExcel)


class TestDataImportErrorDialog:
    def test_base_create(self, qtbot):
        dial = DataImportErrorDialog({"aaaa": [("bbbb", {"__error__": True, "aa": 1})]})
        qtbot.add_widget(dial)
        assert dial.error_view.topLevelItemCount() == 1
        item = dial.error_view.topLevelItem(0)
        assert item.text(0) == "aaaa"
        assert item.childCount() == 1
        assert item.child(0).text(0) == "bbbb"
        assert item.child(0).childCount() == 1

    def test_multiple_errors_one_entry(self, qtbot):
        dial = DataImportErrorDialog(
            {
                "aaaa": [
                    (
                        "bbbb",
                        {
                            "__error__": True,
                            "aa": 1,
                            "ee": {"__error__": True, "eee": 2},
                            "gg": {"__error__": True, "ggg": 3},
                        },
                    )
                ]
            }
        )
        qtbot.addWidget(dial)
        assert dial.error_view.topLevelItemCount() == 1
        item = dial.error_view.topLevelItem(0)
        assert item.childCount() == 1
        assert item.child(0).childCount() == 2

    def test_multiple_entries(self, qtbot):
        dial = DataImportErrorDialog(
            {"aaaa": [("bbbb", {"__error__": True, "aa": 1}), ("cccc", {"__error__": True, "aa": 1})]}
        )
        qtbot.addWidget(dial)
        assert dial.error_view.topLevelItemCount() == 1
        item = dial.error_view.topLevelItem(0)
        assert item.childCount() == 2

    def test_exception_entry(self, qtbot):
        dial = DataImportErrorDialog({"aaaa": ValueError("text")})
        qtbot.addWidget(dial)
        assert dial.error_view.topLevelItemCount() == 1
        item = dial.error_view.topLevelItem(0)
        assert item.childCount() == 1

    def test_clipboard(self, qtbot):
        dial = DataImportErrorDialog({"aaaa": [("bbbb", {"__error__": True, "aa": 1})]})
        qtbot.addWidget(dial)
        dial._copy_to_clipboard()
        text = QApplication.clipboard().text()
        assert text.startswith("aaaa\n")
        assert "__error__" in text
        assert "'aa': 1" in text


class TestMaskWidget:
    def test_create(self, qtbot, part_settings):
        widget = MaskWidget(part_settings)
        qtbot.addWidget(widget)
        assert isinstance(widget.get_mask_property(), MaskProperty)

    def test_update_dilate(self, qtbot, part_settings, image):
        widget = MaskWidget(part_settings)
        qtbot.addWidget(widget)
        assert not widget.dilate_radius.isEnabled()
        with qtbot.waitSignal(widget.values_changed):
            widget.dilate_dim.setCurrentEnum(RadiusType.R2D)
        assert widget.dilate_radius.isEnabled()
        with qtbot.waitSignal(widget.values_changed):
            widget.dilate_radius.setValue(10)
        assert widget.get_dilate_radius() == 10
        with qtbot.waitSignal(widget.values_changed):
            widget.dilate_dim.setCurrentEnum(RadiusType.R3D)
        # image is 2D so radius is still 2D
        assert widget.get_dilate_radius() == 10
        image = image.substitute(image_spacing=[100, 10, 10])
        part_settings.image = image
        # image is 3D so radius is now 3D
        assert widget.get_dilate_radius() == [1, 10, 10]
        with qtbot.waitSignal(widget.values_changed):
            widget.dilate_radius.setValue(11)
        assert widget.get_dilate_radius() == [1, 11, 11]

    @pytest.mark.parametrize(
        "name,func,value",
        [
            ("fill_holes", "setCurrentEnum", RadiusType.R2D),
            ("max_holes_size", "setValue", 10),
            ("save_components", "setChecked", True),
            ("clip_to_mask", "setChecked", True),
            ("reversed_mask", "setChecked", True),
        ],
    )
    def test_update_attribute(self, qtbot, part_settings, name, func, value):
        widget = MaskWidget(part_settings)
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.values_changed):
            widget.fill_holes.setCurrentEnum(RadiusType.R3D)
        with qtbot.waitSignal(widget.values_changed):
            getattr(getattr(widget, name), func)(value)

        assert getattr(widget.get_mask_property(), name) == value

    def test_set_mask_property(self, qtbot, part_settings, mask_property_non_default):
        widget = MaskWidget(part_settings)
        qtbot.addWidget(widget)
        widget.set_mask_property(mask_property_non_default)
        assert widget.get_mask_property() == mask_property_non_default


class TestMaskDialogBase:
    def test_create(self, qtbot, part_settings):
        dialog = MaskDialogBase(part_settings)
        qtbot.addWidget(dialog)
        assert dialog.mask_widget.get_mask_property() == MaskProperty.simple_mask()

    def test_create_mask_property_settings(self, qtbot, part_settings, mask_property_non_default):
        part_settings.set("mask_manager.mask_property", mask_property_non_default)
        dialog = MaskDialogBase(part_settings)
        qtbot.addWidget(dialog)
        assert dialog.mask_widget.get_mask_property() == mask_property_non_default


class TestSpacing:
    def test_create(self, qtbot):
        widget = Spacing(title="Test", data_sequence=(10**-9, 10**-9, 10**-9), unit=Units.nm)

        qtbot.addWidget(widget)

    def test_get_values(self, qtbot):
        widget = Spacing(title="Test", data_sequence=(10**-9, 10**-9, 10**-9), unit=Units.nm)
        qtbot.addWidget(widget)
        assert widget.get_values() == [10**-9, 10**-9, 10**-9]
        assert widget.get_unit_str() == "nm"
        widget.units.setCurrentEnum(Units.µm)
        assert widget.get_values() == [10**-6, 10**-6, 10**-6]


def test_info_label(qtbot, monkeypatch):
    widget = InfoLabel(["Test", "Test2", "Test3"], delay=300)
    monkeypatch.setattr(widget.timer, "start", lambda _: None)
    qtbot.addWidget(widget)
    widget.time = 250
    assert widget.label.text() == "Test"
    qtbot.wait(20)
    widget.time = 500
    widget.one_step()
    assert widget.label.text() == "Test2"
    widget.hide()
    qtbot.wait(30)


@pytest.mark.parametrize("platform,program", [("linux", "xdg-open"), ("linux2", "xdg-open"), ("darwin", "open")])
def test_show_directory_dialog(qtbot, monkeypatch, platform, program):
    called = []

    def _test_arg(li):
        assert li[0] == program
        assert li[1] == "data_path"
        called.append(True)

    dialog = DirectoryDialog("data_path")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(subprocess, "Popen", _test_arg)
    dialog.open_folder()
    assert called == [True]


def test_show_directory_dialog_windows(qtbot, monkeypatch):
    called = []

    def _test_arg(arg):
        assert arg == "data_path"
        called.append(True)

    dialog = DirectoryDialog("data_path", "Additional text")
    qtbot.addWidget(dialog)
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "startfile", _test_arg, raising=False)
    dialog.open_folder()
    assert called == [True]


def test_image_adjustment(qtbot, image):
    dialog = ImageAdjustmentDialog(image)
    qtbot.addWidget(dialog)
    assert dialog.result_val is None
    dialog.process()
    assert isinstance(dialog.result_val, ImageAdjustTuple)


def test_collapsable(qtbot):
    widget = QWidget()
    qtbot.addWidget(widget)
    col = CollapseCheckbox("Test")
    chk = QCheckBox()
    col.add_hide_element(chk)
    layout = QVBoxLayout()
    layout.addWidget(chk)
    layout.addWidget(col)
    widget.setLayout(layout)
    widget.show()
    assert chk.isVisible()
    with qtbot.wait_signal(col.stateChanged):
        col.setChecked(True)
    qtbot.wait(50)
    assert not chk.isVisible()
    with qtbot.wait_signal(col.stateChanged):
        col.setChecked(False)
    qtbot.wait(50)
    assert chk.isVisible()
    col.remove_hide_element(chk)
    with qtbot.wait_signal(col.stateChanged):
        col.setChecked(True)
    assert chk.isVisible()
    widget.hide()


def test_multiple_files_tree_widget(qtbot, monkeypatch):
    from PartSeg.common_gui import multiple_file_widget

    def _monkey_qmenu(func):
        res = QMenu()
        monkeypatch.setattr(res, "exec_", partial(func, res))
        return res

    called = 0

    def _exec_create(num):
        def _exec_(self, _pos):
            assert len(self.actions()) == num

            nonlocal called
            called = num

        return _exec_

    widget = MultipleFilesTreeWidget(compare=True)
    qtbot.addWidget(widget)
    item = QTreeWidgetItem(["file_path"])
    widget.addTopLevelItem(item)
    sub_item2 = QTreeWidgetItem(item, ["raw image"])
    sub_item3 = QTreeWidgetItem(item, ["state2"])
    monkeypatch.setattr(widget, "itemAt", lambda _: item)
    monkeypatch.setattr(widget, "mapToGlobal", lambda _: QPoint(0, 0))
    monkeypatch.setattr(multiple_file_widget, "QMenu", partial(_monkey_qmenu, _exec_create(1)))
    widget.showContextMenu(QPoint(0, 0))
    assert called == 1

    monkeypatch.setattr(widget, "itemAt", lambda _: sub_item2)
    monkeypatch.setattr(multiple_file_widget, "QMenu", partial(_monkey_qmenu, _exec_create(2)))

    widget.showContextMenu(QPoint(0, 0))
    assert called == 2

    monkeypatch.setattr(widget, "itemAt", lambda _: sub_item3)
    monkeypatch.setattr(multiple_file_widget, "QMenu", partial(_monkey_qmenu, _exec_create(3)))

    widget.showContextMenu(QPoint(0, 0))
    assert called == 3


@pytest.mark.parametrize("exc", [MemoryError, IOError, KeyError])
def test_exception_hooks(exc, monkeypatch):
    called = False

    def mock_show_warning(text1, text2):
        nonlocal called
        assert text1 == exception_hooks.OPEN_ERROR
        assert "Test text" in text2
        called = True

    monkeypatch.setattr(exception_hooks, "show_warning", mock_show_warning)
    exception_hooks.load_data_exception_hook(exc("Test text"))
    assert called


def test_exception_hooks_value_error(monkeypatch):
    called = False

    def mock_show_warning(text1, text2):
        nonlocal called
        assert text1 == exception_hooks.OPEN_ERROR
        assert text2 == "Most probably you try to load mask from other image. Check selected files."
        called = True

    monkeypatch.setattr(exception_hooks, "show_warning", mock_show_warning)
    exception_hooks.load_data_exception_hook(ValueError(INCOMPATIBLE_IMAGE_MASK))
    assert called


def test_exception_hooks_other_exception():
    raised = False
    try:
        exception_hooks.load_data_exception_hook(ValueError("Test text"))
    except ValueError:
        raised = True
    finally:
        assert raised


def test_update_dict():
    from PartSeg.common_gui.algorithms_description import recursive_update

    assert recursive_update({}, {}) == {}
    assert recursive_update(None, {}) == {}
    assert recursive_update(None, {"a": 1}) == {"a": 1}
    assert recursive_update({"a": 1}, {"a": 2}) == {"a": 2}
    assert recursive_update({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}
    assert recursive_update({"a": {"a": 1}}, {"a": {"a": 2}}) == {"a": {"a": 2}}
    assert recursive_update({"a": {"a": 1}}, {"a": {"b": 2}}) == {"a": {"a": 1, "b": 2}}


def test_pretty_print():
    from PartSeg.common_gui.algorithms_description import _pretty_print

    assert _pretty_print({"a": 1}) == "\n  a: 1"
    assert _pretty_print({"a": 1}, indent=0) == "\na: 1"
    assert _pretty_print({"a": {"a": 1}}) == "\n  a: \n    a: 1"


def test_profile_select(qtbot, part_settings):
    widget = ProfileSelect()
    qtbot.addWidget(widget)
    assert widget._settings is None
    assert widget.get_value() is None
    widget.set_settings(part_settings)
    assert widget._settings is part_settings
    assert widget.count() == 0
    assert widget.get_value() is None
    part_settings.roi_profiles["test"] = ROIExtractionProfile(name="test", algorithm="test", values={})
    assert widget.count() == 1
    assert widget.get_value() == ROIExtractionProfile(name="test", algorithm="test", values={})
    val2 = ROIExtractionProfile(name="test2", algorithm="test", values={})
    part_settings.roi_profiles["test2"] = val2
    assert widget.count() == 2
    assert widget.currentText() == "test"
    widget.set_value(val2)
    assert widget.currentText() == "test2"


class TestBaseAlgorithmSettingsWidget:
    def test_init(self, qtbot, part_settings):
        widget = BaseAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm)
        qtbot.addWidget(widget)

    @pytest.mark.parametrize(
        "exc, expected",
        [
            (SegmentationLimitException("Test text"), "During segmentation process algorithm meet"),
            (
                RuntimeError("Exception thrown in SimpleITK KittlerIllingworthThreshold\naaa"),
                "Fail to apply Kittler Illingworth to current",
            ),
        ],
    )
    def test_exception_occurred(self, monkeypatch, exc, expected, qtbot):
        from PartSeg.common_gui import algorithms_description

        called = False

        def _exec(self):
            assert self.text().startswith(expected)
            nonlocal called
            called = True

        monkeypatch.setattr(algorithms_description.QMessageBox, "exec_", _exec)
        BaseAlgorithmSettingsWidget.exception_occurred(exc)
        assert called

    def test_exception_occurred_other_exception(self, monkeypatch, qtbot):
        from PartSeg.common_gui import algorithms_description

        called = False

        def _exec(self):
            nonlocal called
            called = True

        monkeypatch.setattr(algorithms_description.ErrorDialog, "exec_", _exec)
        BaseAlgorithmSettingsWidget.exception_occurred(ValueError("Test text"))
        assert called

    def test_show_info(self, qtbot, part_settings):
        widget = BaseAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm)
        qtbot.addWidget(widget)
        widget.show()
        assert not widget.info_label.isVisible()

        widget.show_info("Test text")
        assert widget.info_label.isVisible()
        assert widget.info_label.text() == "Test text"
        widget.show_info("")
        assert not widget.info_label.isVisible()
        assert widget.info_label.text() == ""
        widget.hide()

    def test_image_change(self, qtbot, part_settings, image2):
        widget = BaseAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm)
        qtbot.addWidget(widget)
        assert widget.algorithm_thread.algorithm.image is None
        widget.image_changed(image2)
        assert widget.algorithm_thread.algorithm.image is image2

    def test_mask_change(self, qtbot, part_settings, image2):
        widget = BaseAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm)
        qtbot.addWidget(widget)
        assert widget.mask() is None
        widget.image_changed(image2)
        assert widget.mask() is None
        widget.set_mask(image2.get_channel(0))
        assert widget.mask() is not None

    def test_execute(self, qtbot, part_settings, monkeypatch):
        widget = BaseAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm)
        qtbot.addWidget(widget)
        mock = MagicMock()
        monkeypatch.setattr(widget.algorithm_thread, "start", mock)
        assert part_settings.get(f"algorithms.{LowerThresholdAlgorithm.get_name()}") == {}
        widget.execute()
        mock.assert_called_once()
        assert (
            part_settings.get(f"algorithms.{LowerThresholdAlgorithm.get_name()}")
            == LowerThresholdAlgorithm.__argument_class__()
        )


class TestInteractiveAlgorithmSettingsWidget:
    def test_init(self, qtbot, part_settings):
        widget = InteractiveAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm, [])
        qtbot.addWidget(widget)

    def test_selectors(self, qtbot, part_settings):
        mock1 = MagicMock()
        mock2 = MagicMock()
        widget = InteractiveAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm, [mock1, mock2])
        qtbot.addWidget(widget)
        mock1.setDisabled.assert_not_called()
        widget.disable_selector()
        mock1.setDisabled.assert_called_once()
        mock2.setDisabled.assert_called_once()
        mock1.setEnabled.assert_not_called()
        widget.enable_selector()
        mock1.setEnabled.assert_called_once()
        mock2.setEnabled.assert_called_once()

    def test_set_mask(self, qtbot, part_settings, image, monkeypatch):
        widget = InteractiveAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm, [])
        qtbot.addWidget(widget)
        mock = MagicMock()
        monkeypatch.setattr(widget.algorithm_thread.algorithm, "set_mask", mock)
        with qtbot.wait_signal(part_settings.mask_changed):
            part_settings.mask = image.get_channel(0)
        mock.assert_not_called()
        monkeypatch.setattr(widget, "isVisible", lambda: True)
        with qtbot.wait_signal(part_settings.mask_changed):
            part_settings.mask = image.get_channel(0)
        mock.assert_called_once()
        monkeypatch.undo()

    def test_get_segmentation_profile(self, qtbot, part_settings):
        widget = InteractiveAlgorithmSettingsWidget(part_settings, LowerThresholdAlgorithm, [])
        qtbot.addWidget(widget)
        assert widget.get_segmentation_profile().name == ""
        assert widget.get_segmentation_profile().algorithm == LowerThresholdAlgorithm.get_name()


class TestAlgorithmChooseBase:
    def test_init(self, qtbot, part_settings):
        widget = AlgorithmChooseBase(part_settings, AnalysisAlgorithmSelection)
        qtbot.addWidget(widget)
        assert widget.algorithm_choose.currentText() == AnalysisAlgorithmSelection.get_default().name

    def test_restore_algorithm(self, qtbot, part_settings):
        assert BorderRim.get_name() != AnalysisAlgorithmSelection.get_default().name
        part_settings.set("current_algorithm", BorderRim.get_name())
        widget = AlgorithmChooseBase(part_settings, AnalysisAlgorithmSelection)
        qtbot.addWidget(widget)
        assert widget.algorithm_choose.currentText() == BorderRim.get_name()

    def test_reload(self, qtbot, part_settings):
        # Dummy test to check if code execute
        widget = AlgorithmChooseBase(part_settings, AnalysisAlgorithmSelection)
        qtbot.addWidget(widget)
        widget.reload()


class TestAlgorithmChoose:
    def test_init(self, qtbot, part_settings):
        widget = AlgorithmChoose(part_settings, AnalysisAlgorithmSelection)
        qtbot.addWidget(widget)
        assert widget.algorithm_choose.currentText() == AnalysisAlgorithmSelection.get_default().name

    def test_image_changed(self, qtbot, part_settings, image2, monkeypatch):
        mock = MagicMock()
        widget = AlgorithmChoose(part_settings, AnalysisAlgorithmSelection)
        monkeypatch.setattr(widget.stack_layout.currentWidget(), "image_changed", mock)
        qtbot.addWidget(widget)
        with qtbot.wait_signal(part_settings.image_changed):
            part_settings.image = image2

        mock.assert_called_once()
