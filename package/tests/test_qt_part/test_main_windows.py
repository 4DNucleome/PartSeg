import platform
import sys

import napari
import pytest
import qtpy
from qtpy.QtCore import QCoreApplication

from PartSeg._launcher.main_window import MainWindow as LauncherMainWindow
from PartSeg._roi_analysis import main_window as analysis_main_window
from PartSeg._roi_mask import main_window as mask_main_window

from .utils import CI_BUILD, GITHUB_ACTIONS

napari_warnings = napari.__version__ == "0.3.4" and platform.system() == "Linux" and sys.version_info.minor == 8


def empty(*_):
    pass


class TestAnalysisMainWindow:
    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif(
        (platform.system() == "Windows") and GITHUB_ACTIONS and sys.version_info.minor == 7, reason="need to debug"
    )
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    @pytest.mark.skipif(napari_warnings, reason="warnings fail test")
    def test_opening(self, qtbot, tmpdir):
        main_window = analysis_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()


class TestMaskMainWindow:
    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    @pytest.mark.skipif(napari_warnings, reason="warnings fail test")
    def test_opening(self, qtbot, tmpdir):
        main_window = mask_main_window.MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)


class TestLauncherMainWindow:
    def test_opening(self, qtbot):
        main_window = LauncherMainWindow("Launcher")
        qtbot.addWidget(main_window)

    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_open_mask(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(mask_main_window, "CONFIG_FOLDER", str(tmp_path))
        main_window = LauncherMainWindow("Launcher")
        qtbot.addWidget(main_window)
        main_window._launch_mask()
        with qtbot.waitSignal(main_window.prepare.finished):
            main_window.prepare.start()
        qtbot.addWidget(main_window.wind)
        count = 0
        while main_window.wind.image_view.worker_list:
            if count > 3:
                raise RuntimeError("Problem with clean worker list")
            count += 1
            QCoreApplication.processEvents()

    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    def test_open_analysis(self, qtbot, monkeypatch, tmp_path):
        monkeypatch.setattr(analysis_main_window, "CONFIG_FOLDER", str(tmp_path))
        main_window = LauncherMainWindow("Launcher")
        qtbot.addWidget(main_window)
        main_window._launch_analysis()
        with qtbot.waitSignal(main_window.prepare.finished):
            main_window.prepare.start()
        qtbot.addWidget(main_window.wind)
        count = 0
        while main_window.wind.result_image.worker_list:
            if count > 3:
                raise RuntimeError("Problem with clean worker list")
            count += 1
            QCoreApplication.processEvents()
