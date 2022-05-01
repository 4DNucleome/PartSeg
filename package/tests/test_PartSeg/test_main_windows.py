# pylint: disable=R0201
import platform
import sys
from functools import partial

import pytest
import qtpy
from qtpy.QtCore import QCoreApplication

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


class TestMaskMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        qtbot.wait(50)


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
