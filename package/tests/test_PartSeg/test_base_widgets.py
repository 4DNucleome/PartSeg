# pylint: disable=no-self-use

import os
from unittest.mock import MagicMock

import numpy as np
import pytest
from qtpy.QtCore import QMimeData, QPointF, Qt, QUrl
from qtpy.QtGui import QDropEvent
from qtpy.QtWidgets import QMessageBox

from PartSeg.common_gui.main_window import BaseMainMenu, BaseMainWindow, BaseSettings
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore.analysis import ProjectTuple
from PartSegCore.analysis.load_functions import LoadStackImage, load_dict
from PartSegCore.project_info import AdditionalLayerDescription
from PartSegImage import Image


class TestBaseMainWindow:
    def test_create(self, qtbot, tmp_path):
        main_window = BaseMainWindow(config_folder=tmp_path)
        qtbot.addWidget(main_window)

    def test_lack_of_config_folder(self, qtbot):
        with pytest.raises(ValueError, match="wrong config folder"):
            BaseMainWindow()

    def test_lack_of_config_with_settings(self, qtbot, tmp_path):
        main_window = BaseMainWindow(settings=BaseSettings(tmp_path))
        qtbot.addWidget(main_window)

    def test_toggle_console(self, qtbot, tmp_path):
        main_window = BaseMainWindow(config_folder=tmp_path)
        qtbot.addWidget(main_window)
        main_window.show()
        assert main_window.console is None
        main_window._toggle_console()
        assert main_window.console is not None
        assert main_window.console.isVisible()
        main_window._toggle_console()
        assert main_window.console is not None
        assert not main_window.console.isVisible()

    def test_toggle_multiple_files(self, qtbot, tmp_path):
        main_window = BaseMainWindow(config_folder=tmp_path)
        qtbot.addWidget(main_window)
        main_window.toggle_multiple_files()
        assert main_window.settings.get("multiple_files_widget")
        main_window.toggle_multiple_files()
        assert not main_window.settings.get("multiple_files_widget")

    def test_last_files(self, qtbot, monkeypatch, tmp_path, data_test_dir, part_settings):
        def mock_exec(self):
            self.thread_to_wait.run()
            return True

        monkeypatch.setattr(ExecuteFunctionDialog, "exec_", mock_exec)
        main_window = BaseMainWindow(settings=part_settings, load_dict=load_dict)
        main_window.main_menu = BaseMainMenu(main_window.settings, main_window)
        qtbot.addWidget(main_window)
        assert len(main_window.recent_file_menu.actions()) == 0
        main_window.settings.add_last_files(
            [os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")], LoadStackImage.get_name()
        )
        main_window.settings.add_last_files(
            [os.path.join(data_test_dir, "stack1_components", "stack1_component2.tif")], LoadStackImage.get_name()
        )
        main_window._refresh_recent()
        assert len(main_window.recent_file_menu.actions()) == 2
        assert (
            main_window.recent_file_menu.actions()[0].text()
            == f"{os.path.join(data_test_dir, 'stack1_components', 'stack1_component2.tif')}, "
            f"{LoadStackImage.get_name()}"
        )
        with qtbot.wait_signal(part_settings.image_changed):
            main_window.recent_file_menu.actions()[0].trigger()

    def test_get_colormaps(self, qtbot, part_settings):
        main_window = BaseMainWindow(settings=part_settings)
        qtbot.addWidget(main_window)
        assert len(main_window.get_colormaps()) == part_settings.image.channels

    @pytest.mark.windows_ci_skip
    def test_napari_viewer(self, qtbot, part_settings):
        main_window = BaseMainWindow(settings=part_settings)
        qtbot.addWidget(main_window)
        assert not main_window.viewer_list
        main_window.napari_viewer_show()
        qtbot.wait(50)
        assert len(main_window.viewer_list) == 1
        assert len(main_window.viewer_list[0].layers) == 2
        # with qtbot.wait_signal(main_window.viewer_list[0].window.qt_viewer.destroyed):
        main_window.viewer_list[0].close()
        qtbot.wait(50)
        assert not main_window.viewer_list

    def test_napari_viewer_additional_layers_empty(self, qtbot, part_settings, monkeypatch):
        main_window = BaseMainWindow(settings=part_settings)
        qtbot.addWidget(main_window)
        assert not main_window.viewer_list
        information_mock = MagicMock()
        monkeypatch.setattr(QMessageBox, "information", information_mock)
        main_window.additional_layers_show()
        assert not main_window.viewer_list
        information_mock.assert_called_once()

    @pytest.mark.windows_ci_skip
    @pytest.mark.pyside6_skip
    def test_napari_viewer_additional_layers(self, qtbot, part_settings, monkeypatch):
        main_window = BaseMainWindow(settings=part_settings)
        qtbot.addWidget(main_window)
        assert not main_window.viewer_list
        part_settings._additional_layers = {
            "layer1": AdditionalLayerDescription(np.zeros((10, 10)), "image"),
        }
        main_window.additional_layers_show()
        qtbot.wait(50)
        assert len(main_window.viewer_list) == 1
        assert len(main_window.viewer_list[0].layers) == 1
        # with qtbot.wait_signal(main_window.viewer_list[0].window.qt_viewer.destroyed):
        main_window.viewer_list[0].close()
        assert not main_window.viewer_list
        qtbot.wait(50)
        main_window.additional_layers_show(with_channels=True)
        qtbot.wait(50)
        assert len(main_window.viewer_list) == 1
        assert len(main_window.viewer_list[0].layers) == 3
        # with qtbot.wait_signal(main_window.viewer_list[0].window.qt_viewer.destroyed):
        main_window.viewer_list[0].close()
        qtbot.wait(50)
        assert not main_window.viewer_list

    def test_drop_files(self, qtbot, part_settings, data_test_dir, monkeypatch):
        def mock_exec(self):
            self.thread_to_wait.run()
            return True

        monkeypatch.setattr(ExecuteFunctionDialog, "exec_", mock_exec)

        main_window = BaseMainWindow(settings=part_settings, load_dict=load_dict)
        main_window.main_menu = BaseMainMenu(main_window.settings, main_window)
        qtbot.addWidget(main_window)
        data = QMimeData()
        data.setUrls([QUrl(f"file:/{data_test_dir}/stack1_components/stack1_component1.tif")])

        event = QDropEvent(QPointF(0, 0), Qt.MoveAction, data, Qt.NoButton, Qt.NoModifier)
        main_window.dropEvent(event)


class TestBaseMainMenu:
    def test_create(self, qtbot, tmp_path):
        main_menu = BaseMainMenu(BaseSettings(tmp_path), None)
        qtbot.addWidget(main_menu)

    def test_set_data_list_empty(self, qtbot, monkeypatch):
        warnings_mock = MagicMock()
        main_menu = BaseMainMenu(None, None)
        qtbot.addWidget(main_menu)
        monkeypatch.setattr(QMessageBox, "warning", warnings_mock)
        main_menu.set_data([])
        warnings_mock.assert_called_once()

    def test_none_data(self, qtbot, monkeypatch):
        warnings_mock = MagicMock()
        main_menu = BaseMainMenu(None, None)
        qtbot.addWidget(main_menu)
        monkeypatch.setattr(QMessageBox, "warning", warnings_mock)
        main_menu.set_data(None)
        warnings_mock.assert_called_once()

    def test_set_data_list(self, qtbot, part_settings, image, image2):
        main_window_mock = MagicMock()
        main_menu = BaseMainMenu(part_settings, main_window_mock)
        qtbot.addWidget(main_menu)
        data = [ProjectTuple("path", image), ProjectTuple("path2", image2)]
        main_menu.set_data(data)
        main_window_mock.multiple_files.add_states.assert_called_once()
        main_window_mock.multiple_files.add_states.assert_called_with(data)
        main_window_mock.multiple_files.setVisible.assert_called_with(True)

    def test_swap_time(self, qtbot, part_settings, image, monkeypatch):
        image = image.swap_time_and_stack()
        main_menu = BaseMainMenu(part_settings, None)
        qtbot.addWidget(main_menu)
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.Yes)
        main_menu.set_data(ProjectTuple("path", image))
        assert part_settings.image.times == 1
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.No)
        set_project_info_mock = MagicMock()
        monkeypatch.setattr(part_settings, "set_project_info", set_project_info_mock)
        main_menu.set_data(ProjectTuple("path", image))
        set_project_info_mock.assert_not_called()

    def test_time_and_stack_image(self, qtbot, part_settings, monkeypatch):
        image = Image(np.zeros((10, 10, 10, 10)), image_spacing=(1, 1, 1), axes_order="TZXY")
        main_menu = BaseMainMenu(part_settings, None)
        qtbot.addWidget(main_menu)
        warnings_mock = MagicMock()
        monkeypatch.setattr(QMessageBox, "question", lambda *_: QMessageBox.Yes)
        monkeypatch.setattr(QMessageBox, "warning", warnings_mock)
        main_menu.set_data(ProjectTuple("path", image))
        warnings_mock.assert_called_once()
