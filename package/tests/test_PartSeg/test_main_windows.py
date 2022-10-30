# pylint: disable=R0201
import platform
import sys
from functools import partial
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import qtpy
from qtpy.QtCore import QCoreApplication, Qt

from PartSeg._launcher.main_window import MainWindow as LauncherMainWindow
from PartSeg._launcher.main_window import PartSegGUILauncher
from PartSeg._launcher.main_window import Prepare as LauncherPrepare
from PartSeg._roi_analysis import main_window as analysis_main_window
from PartSeg._roi_mask import main_window as mask_main_window
from PartSegCore import state_store

from .utils import CI_BUILD, GITHUB_ACTIONS, TRAVIS


def empty(*_):
    """To silent some functions"""


pyside_skip = pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")


@pytest.fixture(autouse=True)
def mock_settings_path(tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "save_folder", str(tmp_path))


class TestAnalysisMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="debug test fail")
    @pytest.mark.skipif(
        (platform.system() == "Windows") and GITHUB_ACTIONS and sys.version_info.minor == 7, reason="need to debug"
    )
    @pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = analysis_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()
        main_window.advanced_window.close()
        main_window.advanced_window.close()
        qtbot.wait(50)

    @pyside_skip
    def test_change_theme(self, qtbot, tmpdir):
        main_window = analysis_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.raw_image.viewer.theme == "light"
        main_window.settings.theme_name = "dark"
        assert main_window.raw_image.viewer.theme == "dark"

    @pyside_skip
    def test_scale_bar(self, qtbot, tmpdir):
        main_window = analysis_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window._scale_bar_warning = False
        assert not main_window.result_image.viewer.scale_bar.visible
        main_window._toggle_scale_bar()
        assert main_window.result_image.viewer.scale_bar.visible


@pytest.fixture
def analysis_options(qtbot, part_settings):
    ch_property = analysis_main_window.ChannelProperty(part_settings, "test")
    qtbot.addWidget(ch_property)
    left_image = MagicMock()
    synchronize = MagicMock()
    options = analysis_main_window.Options(part_settings, ch_property, left_image, synchronize)
    qtbot.addWidget(options)
    qtbot.addWidget(options.compare_btn)
    return options


class TestAnalysisOptions:
    def test_create(self, qtbot, analysis_options):

        assert analysis_options.choose_profile.count() == 1
        assert analysis_options.choose_profile.currentText() == "<none>"
        assert analysis_options.choose_pipe.count() == 1
        assert analysis_options.choose_pipe.currentText() == "<none>"

    def test_add_profile(self, qtbot, part_settings, analysis_options, lower_threshold_profile, border_rim_profile):
        part_settings.roi_profiles[lower_threshold_profile.name] = lower_threshold_profile
        assert analysis_options.choose_profile.count() == 2
        assert analysis_options.choose_profile.currentText() == "<none>"
        assert analysis_options.choose_profile.itemText(1) == lower_threshold_profile.name
        assert analysis_options.choose_profile.itemData(1, Qt.ItemDataRole.ToolTipRole) == str(lower_threshold_profile)
        part_settings.roi_profiles[border_rim_profile.name] = border_rim_profile
        assert analysis_options.choose_profile.count() == 3
        assert analysis_options.choose_pipe.count() == 1
        del part_settings.roi_profiles[lower_threshold_profile.name]
        assert analysis_options.choose_profile.count() == 2

    def test_add_pipeline(self, part_settings, analysis_options, sample_pipeline, sample_pipeline2):
        part_settings.roi_pipelines[sample_pipeline.name] = sample_pipeline
        assert analysis_options.choose_pipe.count() == 2
        assert analysis_options.choose_pipe.itemText(1) == sample_pipeline.name
        assert analysis_options.choose_pipe.itemData(1, Qt.ItemDataRole.ToolTipRole) == str(sample_pipeline)
        part_settings.roi_pipelines[sample_pipeline2.name] = sample_pipeline2
        assert analysis_options.choose_pipe.count() == 3
        assert analysis_options.choose_profile.count() == 1

    def test_compare_action(self, part_settings, analysis_options, qtbot):
        roi = np.zeros(part_settings.image.shape, dtype=np.uint8)
        roi[0, 2:10] = 1
        roi[0, 10:-2] = 2
        part_settings.roi = roi
        assert analysis_options.compare_btn.text() == "Compare"
        analysis_options.compare_action()
        assert analysis_options.compare_btn.text() == "Remove"
        assert part_settings.compare_segmentation is not None
        analysis_options.compare_action()
        assert analysis_options.compare_btn.text() == "Compare"
        assert part_settings.compare_segmentation is not None

    @patch("PartSeg._roi_analysis.main_window.QMessageBox.information")
    def test_empty_save_pipeline(self, info, analysis_options, part_settings):
        assert part_settings.history_size() == 0
        analysis_options.save_pipeline()
        info.assert_called_once()
        assert info.call_args[0][1] == "No mask created"

    @patch("PartSeg._roi_analysis.main_window.QMessageBox.information")
    def test_save_pipeline_no_segmentation(self, info, analysis_options, part_settings, history_element):
        part_settings.add_history_element(history_element)
        analysis_options.save_pipeline()
        info.assert_called_once()
        assert info.call_args[0][1] == "No segmentation"

    @patch("PartSeg._roi_analysis.main_window.QMessageBox.information")
    def test_save_pipeline_no_algorithm_values(
        self, info, analysis_options, part_settings, history_element, lower_threshold_profile
    ):
        part_settings.add_history_element(history_element)
        part_settings.last_executed_algorithm = lower_threshold_profile.name
        analysis_options.save_pipeline()
        info.assert_called_once()
        assert info.call_args[0][1] == "Some problem"

    @patch("PartSeg._roi_analysis.main_window.QInputDialog.getText", return_value=("test", True))
    def test_save_pipeline(self, info, analysis_options, part_settings, history_element, lower_threshold_profile):
        part_settings.add_history_element(history_element)
        part_settings.set_algorithm(f"algorithms.{lower_threshold_profile.name}", lower_threshold_profile)
        part_settings.last_executed_algorithm = lower_threshold_profile.name
        assert analysis_options.choose_pipe.count() == 1
        analysis_options.save_pipeline()
        info.assert_called_once()
        assert "test" in part_settings.roi_pipelines
        assert info.call_args[0][1] == "Pipeline name"
        assert analysis_options.choose_pipe.count() == 2
        assert analysis_options.choose_pipe.itemText(1) == "test"

    @patch("PartSeg._roi_analysis.main_window.QInputDialog.getText", return_value=("profile", True))
    def test_save_profile(self, info, analysis_options, part_settings):
        assert analysis_options.choose_profile.count() == 1
        analysis_options.save_profile()
        assert analysis_options.choose_profile.count() == 2
        assert analysis_options.choose_profile.itemText(1) == "profile"
        info.assert_called_once()


class TestMaskMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        qtbot.wait(50)

    @pyside_skip
    def test_change_theme(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.image_view.viewer.theme == "light"
        main_window.settings.theme_name = "dark"
        assert main_window.image_view.viewer.theme == "dark"

    @pyside_skip
    def test_scale_bar(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window._scale_bar_warning = False
        assert not main_window.image_view.viewer.scale_bar.visible
        main_window._toggle_scale_bar()
        assert main_window.image_view.viewer.scale_bar.visible


class TestLauncherMainWindow:
    def test_opening(self, qtbot):
        main_window = LauncherMainWindow("Launcher")
        qtbot.addWidget(main_window)

    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.enablethread
    @pyside_skip
    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_open_mask(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(mask_main_window, "CONFIG_FOLDER", str(tmp_path))
        if platform.system() == "Linux" and (GITHUB_ACTIONS or TRAVIS):
            monkeypatch.setattr(mask_main_window.MainWindow, "show", empty)
        main_window = PartSegGUILauncher()
        qtbot.addWidget(main_window)
        with qtbot.waitSignal(main_window.prepare.finished, timeout=10**4):
            main_window.launch_mask()
        QCoreApplication.processEvents()
        qtbot.add_widget(main_window.wind[0])
        main_window.wind[0].hide()
        qtbot.wait(50)

    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.enablethread
    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    @pyside_skip
    def test_open_analysis(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(analysis_main_window, "CONFIG_FOLDER", str(tmp_path))
        if platform.system() in {"Darwin", "Linux"} and (GITHUB_ACTIONS or TRAVIS):
            monkeypatch.setattr(analysis_main_window.MainWindow, "show", empty)
        main_window = PartSegGUILauncher()
        qtbot.addWidget(main_window)
        with qtbot.waitSignal(main_window.prepare.finished):
            main_window.launch_analysis()
        QCoreApplication.processEvents()
        qtbot.wait(50)
        qtbot.add_widget(main_window.wind[0])
        main_window.wind[0].hide()
        qtbot.wait(50)

    def test_prepare(self):
        prepare = LauncherPrepare("PartSeg._roi_analysis.main_window")
        prepare.run()
        assert isinstance(prepare.result, partial)
        assert prepare.result.func is analysis_main_window.MainWindow
