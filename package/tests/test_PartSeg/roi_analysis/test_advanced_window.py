# pylint: disable=no-self-use
from unittest.mock import Mock

from PartSeg._roi_analysis.advanced_window import MeasurementSettings, MultipleInput, Properties
from PartSeg._roi_analysis.advanced_window import QInputDialog as advanced_module_input
from PartSeg._roi_analysis.advanced_window import QMessageBox as advanced_message_box
from PartSegCore.analysis import AnalysisAlgorithmSelection


class TestProperties:
    def test_synchronize_voxel_size(self, qtbot, part_settings):
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        widget.lock_spacing.setChecked(True)
        widget.update_spacing()
        value = widget.spacing[1].value()
        with qtbot.waitSignal(widget.spacing[2].valueChanged, timeout=10**4):
            widget.spacing[2].setValue(value - 20)
        assert widget.spacing[1].value() == value - 20

    def test_pipeline_profile_show_info(
        self, qtbot, part_settings, border_rim_profile, lower_threshold_profile, sample_pipeline
    ):
        part_settings.roi_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.roi_pipelines[sample_pipeline.name] = sample_pipeline
        widget = Properties(part_settings)
        widget.show()
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 1
        part_settings.roi_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        assert widget.pipeline_list.count() == 1
        assert widget.info_label.toPlainText() == ""
        with qtbot.waitSignal(widget.profile_list.currentItemChanged, timeout=10**4):
            widget.profile_list.setCurrentRow(1)
        profile = part_settings.roi_profiles[widget.profile_list.item(1).text()]
        assert widget.info_label.toPlainText() == profile.pretty_print(AnalysisAlgorithmSelection)
        widget.pipeline_list.setCurrentRow(0)
        assert widget.info_label.toPlainText() == sample_pipeline.pretty_print(AnalysisAlgorithmSelection)
        widget.hide()

    def test_delete_profile(self, qtbot, part_settings, border_rim_profile, lower_threshold_profile):
        part_settings.roi_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.roi_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget = Properties(part_settings)
        widget.show()
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        with qtbot.waitSignal(widget.profile_list.currentItemChanged, timeout=10**4):
            widget.profile_list.setCurrentRow(0)
        assert widget.delete_btn.isEnabled()
        with qtbot.waitSignal(widget.delete_btn.clicked):
            widget.delete_btn.click()
        assert len(part_settings.roi_profiles) == 1
        assert lower_threshold_profile.name in part_settings.roi_profiles
        widget.hide()

    def test_rename_profile(
        self, qtbot, part_settings, border_rim_profile, lower_threshold_profile, sample_pipeline, monkeypatch
    ):
        part_settings.roi_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.roi_pipelines[sample_pipeline.name] = sample_pipeline
        part_settings.roi_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget = Properties(part_settings)
        widget.show()
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        monkeypatch.setattr(advanced_module_input, "getText", check_text(border_rim_profile.name, "rim"))
        widget.profile_list.setCurrentRow(0)
        widget.rename_profile()
        assert widget.profile_list.item(1).text() == "rim"
        assert set(part_settings.roi_profiles.keys()) == {"rim", lower_threshold_profile.name}
        monkeypatch.setattr(advanced_module_input, "getText", check_text(sample_pipeline.name, "rim"))
        widget.pipeline_list.setCurrentRow(0)
        widget.rename_profile()
        assert widget.pipeline_list.item(0).text() == "rim"
        assert set(part_settings.roi_pipelines.keys()) == {"rim"}
        monkeypatch.setattr(advanced_module_input, "getText", check_text("rim", lower_threshold_profile.name))

        called_mock = [0]

        def mock_waring(*_):
            monkeypatch.setattr(widget, "rename_profile", _empty)
            called_mock[0] = 1
            return advanced_message_box.No

        monkeypatch.setattr(advanced_message_box, "warning", mock_waring)
        widget.profile_list.setCurrentRow(1)
        widget.rename_profile()
        assert called_mock[0] == 1
        assert widget.profile_list.item(1).text() == "rim"
        assert set(part_settings.roi_profiles.keys()) == {"rim", lower_threshold_profile.name}
        widget.hide()

    def test_multiple_files_visibility(self, qtbot, part_settings):
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        assert not part_settings.get("multiple_files_widget")
        assert not widget.multiple_files_chk.isChecked()
        with qtbot.waitSignal(widget.multiple_files_chk.stateChanged):
            widget.multiple_files_chk.setChecked(True)
        assert part_settings.get("multiple_files_widget")
        part_settings.set("multiple_files_widget", False)
        assert not widget.multiple_files_chk.isChecked()


class TestMeasurementSettings:
    def test_create(self, qtbot, part_settings):
        widget = MeasurementSettings(part_settings)
        qtbot.addWidget(widget)

    def test_base_steep(self, qtbot, part_settings):
        widget = MeasurementSettings(part_settings)
        qtbot.addWidget(widget)
        widget.show()
        widget.profile_options.setCurrentRow(0)
        assert widget.profile_options.item(0).text() == "Volume"
        assert widget.profile_options.item(1).text() == "Diameter"
        assert widget.profile_options_chosen.count() == 0
        widget.choose_option()
        assert widget.profile_options_chosen.count() == 1
        widget.choose_option()
        assert widget.profile_options_chosen.count() == 1
        widget.profile_options.setCurrentRow(1)
        assert widget.profile_options_chosen.count() == 1
        widget.choose_option()
        assert widget.profile_options_chosen.count() == 2
        widget.profile_options.setCurrentRow(0)
        widget.proportion_action()
        assert widget.profile_options_chosen.count() == 2
        widget.profile_options.setCurrentRow(1)
        widget.proportion_action()
        assert widget.profile_options_chosen.count() == 3
        assert widget.profile_options_chosen.item(2).text() == "ROI Volume/ROI Diameter"

        widget.profile_options_chosen.setCurrentRow(0)
        assert widget.profile_options_chosen.item(0).text() == "ROI Volume"
        widget.remove_element()
        assert widget.profile_options_chosen.count() == 2
        assert widget.profile_options_chosen.item(0).text() == "ROI Diameter"

        assert not widget.save_butt.isEnabled()
        with qtbot.waitSignal(widget.profile_name.textChanged):
            widget.profile_name.setText("test")
        assert widget.save_butt.isEnabled()

        assert len(part_settings.measurement_profiles) == 3
        with qtbot.waitSignal(widget.save_butt.clicked):
            widget.save_butt.click()
        assert len(part_settings.measurement_profiles) == 4

        with qtbot.waitSignal(widget.profile_name.textChanged):
            widget.profile_name.setText("")
        assert not widget.save_butt.isEnabled()

        widget.reset_action()
        assert widget.profile_options_chosen.count() == 0
        widget.hide()


def test_multiple_input(qtbot, monkeypatch):
    mock = Mock()
    monkeypatch.setattr(advanced_message_box, "warning", mock)
    widget = MultipleInput(
        text="sample text",
        help_text="help",
        objects_list=[
            ("A", str),
            ("B", int, 5),
            ("C", float, 5.0),
            ("D", int),
            ("E", float),
        ],
    )
    qtbot.addWidget(widget)
    assert widget.result is None
    widget.accept_response()
    assert widget.result is None
    mock.assert_called_once()
    mock.reset_mock()
    widget.object_dict["A"][1].setText("test")
    widget.accept_response()
    mock.assert_not_called()
    assert widget.result == {"A": "test", "B": 5, "C": 5.0, "D": 0, "E": 0.0}


def check_text(expected, to_return):
    def _check(*_, text=None, **_kwargs):
        assert text == expected
        return to_return, True

    return _check


def _empty():
    """
    Empty function for monkeypatching to prevent recursive call
    in `test_rename_profile` test
    """
