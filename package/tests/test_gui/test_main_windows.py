from PartSeg.segmentation_analysis.main_window import MainWindow as AnalysisMainWindow
from PartSeg.segmentation_mask.stack_gui_main import MainWindow as MaskMainWindow


class TestAnalysisMainWindow:
    def test_opening(self, qtbot, tmpdir):
        main_window = AnalysisMainWindow(tmpdir)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()


class TestMaskMainWindow:
    def test_opening(self, qtbot, tmpdir):
        main_window = MaskMainWindow(tmpdir)
        qtbot.addWidget(main_window)
