# pylint: disable=no-self-use
import os
import platform
from functools import partial

import pytest
from qtpy.QtCore import QCoreApplication

from PartSeg._launcher.main_window import MainWindow as LauncherMainWindow
from PartSeg._launcher.main_window import PartSegGUILauncher
from PartSeg._launcher.main_window import Prepare as LauncherPrepare
from PartSeg._roi_analysis import main_window as analysis_main_window
from PartSeg._roi_mask import main_window as mask_main_window

GITHUB_ACTIONS = "GITHUB_ACTIONS" in os.environ


def empty(*_):
    """To silent some functions"""


class TestLauncherMainWindow:
    def test_opening(self, qtbot):
        main_window = LauncherMainWindow("Launcher")
        qtbot.addWidget(main_window)

    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.enablethread
    @pytest.mark.pyside_skip
    @pytest.mark.windows_ci_skip
    def test_open_mask(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(mask_main_window, "CONFIG_FOLDER", str(tmp_path))
        if platform.system() == "Linux" and GITHUB_ACTIONS:
            monkeypatch.setattr(mask_main_window.MainWindow, "show", empty)
        main_window = PartSegGUILauncher()
        qtbot.addWidget(main_window)
        with qtbot.waitSignal(main_window.prepare.finished, timeout=10**4):
            main_window.launch_mask()
        QCoreApplication.instance().processEvents()
        qtbot.add_widget(main_window.wind[0])
        main_window.wind[0].hide()
        qtbot.wait(50)

    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.enablethread
    @pytest.mark.windows_ci_skip
    @pytest.mark.pyside_skip
    def test_open_analysis(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(analysis_main_window, "CONFIG_FOLDER", str(tmp_path))
        if platform.system() in {"Darwin", "Linux"} and GITHUB_ACTIONS:
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
