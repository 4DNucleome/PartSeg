import platform

import pytest
import qtpy

from PartSeg.segmentation_analysis.main_window import MainWindow as AnalysisMainWindow
from PartSeg.segmentation_mask.main_window import MainWindow as MaskMainWindow


class TestAnalysisMainWindow:
    @pytest.mark.skipif(platform.system() == "Windows", reason="vispy problem")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    def test_opening(self, qtbot, tmpdir):
        main_window = AnalysisMainWindow(tmpdir)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()


class TestMaskMainWindow:
    @pytest.mark.skipif(platform.system() == "Windows", reason="vispy problem")
    @pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem")
    def test_opening(self, qtbot, tmpdir):
        main_window = MaskMainWindow(tmpdir)
        qtbot.addWidget(main_window)
