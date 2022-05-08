import os

import pytest

from PartSeg.common_gui.main_window import BaseMainMenu, BaseMainWindow, BaseSettings
from PartSeg.common_gui.waiting_dialog import ExecuteFunctionDialog
from PartSegCore.analysis.load_functions import LoadStackImage, load_dict


class TestBaseMainWindow:
    def test_create(self, qtbot, tmp_path):
        main_window = BaseMainWindow(config_folder=tmp_path)
        qtbot.addWidget(main_window)

    def test_lack_of_config_folder(self, qtbot):
        with pytest.raises(ValueError):
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


class TestBaseMainMenu:
    def test_create(self, qtbot, tmp_path):
        main_menu = BaseMainMenu(BaseSettings(tmp_path), None)
        qtbot.addWidget(main_menu)
