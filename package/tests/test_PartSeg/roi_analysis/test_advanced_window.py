from PartSeg._roi_analysis.advanced_window import Properties
from PartSeg._roi_analysis.advanced_window import QInputDialog as advanced_module_input
from PartSeg._roi_analysis.advanced_window import QMessageBox as advanced_message_box
from PartSegCore.analysis import analysis_algorithm_dict


class TestProperties:
    def test_synchronize_voxel_size(self, qtbot, part_settings):
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        widget.lock_spacing.setChecked(True)
        widget.update_spacing()
        value = widget.spacing[1].value()
        with qtbot.waitSignal(widget.spacing[2].valueChanged, timeout=10 ** 4):
            widget.spacing[2].setValue(value - 20)
        assert widget.spacing[1].value() == value - 20

    def test_pipeline_profile_show_info(
        self, qtbot, part_settings, border_rim_profile, lower_threshold_profile, sample_pipeline
    ):
        part_settings.segmentation_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.segmentation_pipelines[sample_pipeline.name] = sample_pipeline
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 1
        part_settings.segmentation_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        assert widget.pipeline_list.count() == 1
        assert widget.info_label.toPlainText() == ""
        with qtbot.waitSignal(widget.profile_list.currentItemChanged, timeout=10 ** 4):
            widget.profile_list.setCurrentRow(1)
        profile = part_settings.segmentation_profiles[widget.profile_list.item(1).text()]
        assert widget.info_label.toPlainText() == profile.pretty_print(analysis_algorithm_dict)
        widget.pipeline_list.setCurrentRow(0)
        assert widget.info_label.toPlainText() == sample_pipeline.pretty_print(analysis_algorithm_dict)

    def test_delete_profile(self, qtbot, part_settings, border_rim_profile, lower_threshold_profile):
        part_settings.segmentation_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.segmentation_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        with qtbot.waitSignal(widget.profile_list.currentItemChanged, timeout=10 ** 4):
            widget.profile_list.setCurrentRow(0)
        assert widget.delete_btn.isEnabled()
        with qtbot.waitSignal(widget.delete_btn.clicked):
            widget.delete_btn.click()
        assert len(part_settings.segmentation_profiles) == 1
        assert lower_threshold_profile.name in part_settings.segmentation_profiles

    def test_rename_profile(
        self, qtbot, part_settings, border_rim_profile, lower_threshold_profile, sample_pipeline, monkeypatch
    ):
        part_settings.segmentation_profiles[border_rim_profile.name] = border_rim_profile
        part_settings.segmentation_pipelines[sample_pipeline.name] = sample_pipeline
        part_settings.segmentation_profiles[lower_threshold_profile.name] = lower_threshold_profile
        widget = Properties(part_settings)
        qtbot.addWidget(widget)
        widget.update_profile_list()
        assert widget.profile_list.count() == 2
        monkeypatch.setattr(advanced_module_input, "getText", check_text(border_rim_profile.name, "rim"))
        widget.profile_list.setCurrentRow(0)
        widget.rename_profile()
        assert widget.profile_list.item(1).text() == "rim"
        assert set(part_settings.segmentation_profiles.keys()) == {"rim", lower_threshold_profile.name}
        monkeypatch.setattr(advanced_module_input, "getText", check_text(sample_pipeline.name, "rim"))
        widget.pipeline_list.setCurrentRow(0)
        widget.rename_profile()
        assert widget.pipeline_list.item(0).text() == "rim"
        assert set(part_settings.segmentation_pipelines.keys()) == {"rim"}
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
        assert set(part_settings.segmentation_profiles.keys()) == {"rim", lower_threshold_profile.name}


def check_text(expected, to_return):
    def _check(*_, text=None, **_kwargs):
        assert text == expected
        return to_return, True

    return _check


def _empty():
    pass
