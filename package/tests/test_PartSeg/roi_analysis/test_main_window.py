from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from qtpy.QtCore import Qt

from PartSeg._roi_analysis.main_window import ChannelProperty, MainWindow, Options
from PartSegCore.analysis import ProjectTuple
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation import ROIExtractionResult


class TestAnalysisMainWindow:
    # @pytest.mark.skipif((platform.system() == "Linux") and CI_BUILD, reason="debug test fail")
    @pytest.mark.pyside_skip
    def test_opening(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window.main_menu.batch_processing_btn.click()
        main_window.main_menu.advanced_btn.click()
        main_window.advanced_window.close()
        main_window.advanced_window.close()
        qtbot.wait(50)

    @pytest.mark.pyside_skip
    def test_change_theme(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        assert main_window.raw_image.viewer.theme == "light"
        main_window.settings.theme_name = "dark"
        assert main_window.raw_image.viewer.theme == "dark"

    @pytest.mark.pyside_skip
    def test_scale_bar(self, qtbot, tmpdir):
        main_window = MainWindow(tmpdir, initial_image=False)
        qtbot.addWidget(main_window)
        main_window._scale_bar_warning = False
        assert not main_window.result_image.viewer.scale_bar.visible
        main_window._toggle_scale_bar()
        assert main_window.result_image.viewer.scale_bar.visible

    def test_get_project_info(self, image, tmp_path):
        res = MainWindow.get_project_info(str(tmp_path / "test.tiff"), image)
        assert isinstance(res.roi_info, ROIInfo)
        assert isinstance(res, ProjectTuple)

        roi = np.zeros(image.shape, dtype=np.uint8)
        roi[:, 2:-2] = 1
        res = MainWindow.get_project_info(str(tmp_path / "test.tiff"), image, ROIInfo(roi))
        assert isinstance(res.roi_info, ROIInfo)
        assert set(res.roi_info.bound_info) == {1}


@pytest.fixture
def analysis_options(qtbot, part_settings):
    ch_property = ChannelProperty(part_settings, "test")
    qtbot.addWidget(ch_property)
    left_image = MagicMock()
    synchronize = MagicMock()
    options = Options(part_settings, ch_property, left_image, synchronize)
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

    def test_execution_done(self, analysis_options, part_settings, monkeypatch):
        sender_mock = MagicMock()
        sender_mock.get_info_text.return_value = "test"
        mock = Mock(return_value=sender_mock)
        info_mock = Mock()

        monkeypatch.setattr(analysis_options, "sender", mock)
        monkeypatch.setattr("qtpy.QtWidgets.QMessageBox.information", info_mock)

        res = ROIExtractionResult(
            roi=np.zeros(part_settings.image.shape, dtype="uint8"),
            parameters=analysis_options.algorithm_choose_widget.algorithm_dict[
                "Lower threshold"
            ].get_segmentation_profile(),
            info_text="info",
        )
        analysis_options.execution_done(res)

        assert analysis_options.label.toPlainText() == "test"
        info_mock.assert_called_once_with(analysis_options, "Algorithm info", "info")
