import platform
import sys

import napari
import pytest
import qtpy

from PartSeg.segmentation_analysis.main_window import MainWindow as AnalysisMainWindow
from PartSeg.segmentation_mask.main_window import MainWindow as MaskMainWindow

from .utils import CI_BUILD, GITHUB_ACTIONS

napari_warnings = napari.__version__ == "0.3.4" and platform.system() == "Linux" and sys.version_info.minor == 8


class TestAnalysisMainWindow:
    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif(
        (platform.system() == "Windows") and GITHUB_ACTIONS and sys.version_info.minor == 7, reason="need to debug"
    )
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    @pytest.mark.skipif(napari_warnings, reason="warnings fail test")
    def test_opening(self, qtbot, tmpdir):
        main_window = AnalysisMainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()


class TestMaskMainWindow:
    @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="vispy problem")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    @pytest.mark.skipif(napari_warnings, reason="warnings fail test")
    def test_opening(self, qtbot, tmpdir):
        main_window = MaskMainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
