# pylint: disable=R0201
from copy import copy

import pytest

from PartSeg._roi_analysis import prepare_plan_widget
from PartSegCore import Units
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import AnalysisAlgorithmSelection, SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.analysis.calculation_plan import CalculationTree, MeasurementCalculate, NodeType, RootType
from PartSegCore.analysis.save_functions import SaveAsTiff
from PartSegCore.io_utils import SaveMaskAsTiff
from PartSegCore.mask_create import MaskProperty
from PartSegCore.segmentation.restartable_segmentation_algorithms import (
    LowerThresholdAlgorithm,
    UpperThresholdAlgorithm,
)


@pytest.mark.parametrize(
    "mask1,mask2,enabled",
    [
        ("mask1", "mask2", True),
        ("mask1", "", False),
        ("", "mask2", False),
        ("", "", False),
        ("mask1", "mask1", False),
        ("mask2", "mask2", False),
        ("mask1", "mask2", True),
        ("mask2", "mask1", True),
        ("mask", "mask2", False),
        ("mask2", "mask", False),
        ("mask", "mask1", False),
        ("mask", "mask2", False),
        ("mask1", "mask", False),
        ("mask2", "mask", False),
    ],
)
def test_two_mask_dialog(qtbot, mask1, mask2, enabled):
    dialog = prepare_plan_widget.TwoMaskDialog(["mask1", "mask2"])
    qtbot.addWidget(dialog)
    assert not dialog.ok_btn.isEnabled()
    dialog.mask1_name.setText(mask1)
    dialog.mask2_name.setText(mask2)
    assert dialog.ok_btn.isEnabled() is enabled
    assert dialog.get_result() == (mask1, mask2)


def test_two_mask_dialog_strip(qtbot):
    dialog = prepare_plan_widget.TwoMaskDialog(["mask1", "mask2"])
    qtbot.addWidget(dialog)
    dialog.mask1_name.setText("mask1 ")
    dialog.mask2_name.setText(" mask2")
    assert dialog.get_result() == ("mask1", "mask2")


@pytest.mark.parametrize(
    "mask,enabled",
    [
        ("mask", False),
        ("", False),
        ("mask1", True),
        ("mask2", True),
    ],
)
def test_mask_dialog(qtbot, mask, enabled):
    dialog = prepare_plan_widget.MaskDialog(["mask1", "mask2"])
    qtbot.addWidget(dialog)
    assert not dialog.ok_btn.isEnabled()
    dialog.mask1_name.setText(mask)
    assert dialog.ok_btn.isEnabled() is enabled
    assert dialog.get_result() == (mask,)


def test_mask_dialog_strip(qtbot):
    dialog = prepare_plan_widget.MaskDialog(["mask1", "mask2"])
    qtbot.addWidget(dialog)
    dialog.mask1_name.setText("mask1 ")
    assert dialog.get_result() == ("mask1",)


class TestFileMaskWidget:
    def test_create(self, qtbot):
        widget = prepare_plan_widget.FileMask()
        qtbot.addWidget(widget)
        assert widget.select_type.currentText() == "Suffix"

    @pytest.mark.parametrize("name", ["", "mask"])
    def test_suffix_state(self, qtbot, name):
        widget = prepare_plan_widget.FileMask()
        qtbot.addWidget(widget)
        res = widget.get_value(name=name)
        assert isinstance(res, prepare_plan_widget.MaskSuffix)
        assert res.name == name
        assert res.suffix == "_mask"
        assert widget.is_valid()

    @pytest.mark.parametrize("name", ["", "mask"])
    def test_replace_state(self, qtbot, name):
        widget = prepare_plan_widget.FileMask()
        widget.show()
        qtbot.addWidget(widget)
        assert not widget.second_text.isVisible()
        with qtbot.waitSignal(widget.select_type.currentIndexChanged):
            widget.select_type.setCurrentIndex(1)
        widget.first_text.setText("val")
        widget.second_text.setText("val2")
        assert widget.second_text.isVisible()
        res = widget.get_value(name=name)
        assert isinstance(res, prepare_plan_widget.MaskSub)
        assert res.name == name
        assert res.base == "val"
        assert res.rep == "val2"
        assert widget.is_valid()
        widget.hide()

    @pytest.mark.parametrize("name", ["", "mask"])
    def test_mapping_state(self, qtbot, name, tmp_path):
        file_path = str(tmp_path / "file.txt")
        with open(file_path, "w") as f:
            f.write("test")

        widget = prepare_plan_widget.FileMask()
        widget.show()
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.select_type.currentIndexChanged):
            widget.select_type.setCurrentIndex(2)
        widget.first_text.setText(file_path)
        res = widget.get_value(name=name)
        assert isinstance(res, prepare_plan_widget.MaskFile)
        assert res.name == name
        assert res.path_to_file == file_path
        assert widget.is_valid()

    def test_change_state_memory(self, qtbot):
        widget = prepare_plan_widget.FileMask()
        qtbot.addWidget(widget)
        widget.change_type(0)
        widget.first_text.setText("suffix_text")
        widget.change_type(1)
        widget.first_text.setText("replace_base")
        widget.second_text.setText("replace")
        widget.change_type(2)
        widget.first_text.setText("file_path")
        widget.change_type(0)
        assert widget.first_text.text() == "suffix_text"
        widget.change_type(1)
        assert widget.first_text.text() == "replace_base"
        assert widget.second_text.text() == "replace"
        widget.change_type(2)
        assert widget.first_text.text() == "file_path"

    def test_select_file(self, qtbot, monkeypatch):
        widget = prepare_plan_widget.FileMask()
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.select_type.currentIndexChanged):
            widget.select_type.setCurrentIndex(2)
        monkeypatch.setattr(prepare_plan_widget.QFileDialog, "exec_", lambda x: True)
        monkeypatch.setattr(prepare_plan_widget.QFileDialog, "selectedFiles", lambda x: ["file_path"])
        assert widget.first_text.text() == ""
        widget.select_file()
        assert widget.first_text.text() == "file_path"


@pytest.fixture
def calculation_plan(measurement_profiles):
    roi_extraction = AnalysisAlgorithmSelection.get_default()
    return prepare_plan_widget.CalculationPlan(
        tree=CalculationTree(
            operation=RootType.Image,
            children=[
                CalculationTree(
                    operation=ROIExtractionProfile(
                        name="test",
                        algorithm=roi_extraction.name,
                        values=roi_extraction.values,
                    ),
                    children=[
                        CalculationTree(
                            operation=MeasurementCalculate(
                                channel=0, units=Units.nm, name_prefix="", measurement_profile=measurement_profiles[0]
                            ),
                            children=[],
                        )
                    ],
                ),
            ],
        ),
        name="test",
    )


class TestCalculateInfo:
    def test_create(self, qtbot, part_settings, calculation_plan):
        part_settings.batch_plans["test"] = calculation_plan
        widget = prepare_plan_widget.CalculateInfo(part_settings)
        qtbot.addWidget(widget)
        assert widget.calculate_plans.count() == 1

    def test_select_plan(self, qtbot, part_settings, calculation_plan):
        part_settings.batch_plans["test"] = calculation_plan
        widget = prepare_plan_widget.CalculateInfo(part_settings)
        qtbot.addWidget(widget)
        assert widget.plan_view.topLevelItemCount() == 0
        with qtbot.waitSignal(widget.calculate_plans.currentRowChanged):
            widget.calculate_plans.setCurrentRow(0)
        assert widget.plan_view.topLevelItemCount() == 1

    def test_add_calculation_plan(self, qtbot, part_settings, calculation_plan):
        part_settings.batch_plans["test"] = calculation_plan
        widget = prepare_plan_widget.CalculateInfo(part_settings)
        qtbot.addWidget(widget)
        assert widget.calculate_plans.count() == 1
        widget.calculate_plans.setCurrentRow(0)
        calculation_plan2 = prepare_plan_widget.CalculationPlan(
            tree=calculation_plan.execution_tree,
            name="test2",
        )
        part_settings.batch_plans["test2"] = calculation_plan2
        assert widget.calculate_plans.count() == 2

    def test_delete_plan(self, qtbot, part_settings, calculation_plan):
        part_settings.batch_plans["test"] = calculation_plan
        widget = prepare_plan_widget.CalculateInfo(part_settings)
        qtbot.addWidget(widget)
        assert widget.calculate_plans.count() == 1
        widget.delete_plan()
        assert widget.calculate_plans.count() == 1
        widget.calculate_plans.setCurrentRow(0)
        widget.delete_plan()
        assert widget.calculate_plans.count() == 0
        assert "test" not in part_settings.batch_plans

    def test_edit_plan(self, qtbot, part_settings, calculation_plan):
        part_settings.batch_plans["test"] = calculation_plan
        widget = prepare_plan_widget.CalculateInfo(part_settings)
        qtbot.addWidget(widget)
        with qtbot.assert_not_emitted(widget.plan_to_edit_signal):
            widget.edit_plan()
        widget.calculate_plans.setCurrentRow(0)
        with qtbot.waitSignal(widget.plan_to_edit_signal):
            widget.edit_plan()
        assert widget.plan_to_edit.name == "test"


class TestOtherOperations:
    def test_create(self, qtbot):
        widget = prepare_plan_widget.OtherOperations()
        qtbot.addWidget(widget)

    def test_change_root_type(self, qtbot):
        widget = prepare_plan_widget.OtherOperations()
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.root_type_changed, check_params_cb=lambda x: x == RootType.Project):
            widget.change_root.setCurrentEnum(RootType.Project)

    def test_update_save_combo(self, qtbot):
        widget = prepare_plan_widget.OtherOperations()
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.save_choose.currentTextChanged):
            widget.save_choose.setCurrentIndex(1)
        assert widget.save_btn.text().startswith("Save to")

        widget.save_changed("eeee")
        assert widget.save_btn.text() == "Save"
        assert widget.save_choose.currentIndex() == 0

    def test_set_current_node(self, qtbot):
        widget = prepare_plan_widget.OtherOperations()
        qtbot.addWidget(widget)
        widget.set_current_node(None)
        assert not widget.save_btn.isEnabled()
        widget.set_current_node(NodeType.root)
        assert not widget.save_btn.isEnabled()
        with qtbot.waitSignal(widget.save_choose.currentTextChanged):
            widget.save_choose.setCurrentText(SaveAsTiff.get_short_name())
        assert widget.save_btn.isEnabled()
        with qtbot.waitSignal(widget.save_choose.currentTextChanged):
            widget.save_choose.setCurrentText(SaveMaskAsTiff.get_short_name())
        assert not widget.save_btn.isEnabled()

    def test_save_action(self, qtbot, monkeypatch):
        monkeypatch.setattr(prepare_plan_widget.FormDialog, "exec_", lambda x: True)
        monkeypatch.setattr(
            prepare_plan_widget.FormDialog,
            "get_values",
            lambda x: {"suffix": "test", "directory": "test2", "sample": 1},
        )

        def check_save_params(params):
            assert params.suffix == "test"
            assert params.directory == "test2"
            assert params.algorithm == SaveAsTiff.get_name()
            assert params.short_name == SaveAsTiff.get_short_name()
            assert params.values == {"sample": 1}
            return True

        widget = prepare_plan_widget.OtherOperations()
        qtbot.addWidget(widget)
        widget.set_current_node(NodeType.root)
        with qtbot.waitSignal(widget.save_choose.currentTextChanged):
            widget.save_choose.setCurrentText(SaveAsTiff.get_short_name())
        with qtbot.waitSignal(widget.save_operation, check_params_cb=check_save_params):
            widget.save_btn.click()


class TestROIExtraction:
    def test_create(self, qtbot, part_settings):
        widget = prepare_plan_widget.ROIExtraction(settings=part_settings)
        qtbot.addWidget(widget)

    def test_selected_profile(self, qtbot, part_settings):
        def check_profile(name):
            def _check_profile(profile: ROIExtractionProfile):
                assert profile == part_settings.roi_profiles[name]
                return True

            return _check_profile

        part_settings.roi_profiles["test"] = ROIExtractionProfile(
            name="test",
            algorithm=LowerThresholdAlgorithm.get_name(),
            values=LowerThresholdAlgorithm.get_default_values(),
        )
        part_settings.roi_profiles["test2"] = ROIExtractionProfile(
            name="test2",
            algorithm=UpperThresholdAlgorithm.get_name(),
            values=UpperThresholdAlgorithm.get_default_values(),
        )
        widget = prepare_plan_widget.ROIExtraction(settings=part_settings)
        qtbot.addWidget(widget)

        assert widget.roi_extraction_profile.count() == 2
        with qtbot.assert_not_emitted(widget.roi_extraction_profile_add):
            widget._add_profile()
        widget.segment_stack.setCurrentWidget(widget.roi_extraction_profile)
        with qtbot.waitSignal(widget.roi_extraction_profile_selected, check_params_cb=check_profile("test")):
            widget.roi_extraction_profile.setCurrentRow(0)

        with qtbot.waitSignal(widget.roi_extraction_profile_selected, check_params_cb=check_profile("test2")):
            widget.roi_extraction_profile.setCurrentRow(1)

        with widget.enable_protect():
            with qtbot.assert_not_emitted(widget.roi_extraction_profile_selected):
                widget.roi_extraction_profile.setCurrentRow(0)

        part_settings.roi_profiles["test3"] = ROIExtractionProfile(
            name="test3",
            algorithm=LowerThresholdAlgorithm.get_name(),
            values=LowerThresholdAlgorithm.get_default_values(),
        )
        assert widget.roi_extraction_profile.count() == 3

        with qtbot.waitSignal(widget.roi_extraction_profile_add, check_params_cb=check_profile("test")):
            widget._add_profile()

    def test_select_pipeline(self, qtbot, part_settings):
        def check_pipeline(name):
            def _check_pipeline(pipeline: SegmentationPipeline):
                assert pipeline == part_settings.roi_pipelines[name]
                return True

            return _check_pipeline

        part_settings.roi_pipelines["test"] = SegmentationPipeline(
            name="test",
            segmentation=ROIExtractionProfile(
                name="test3",
                algorithm=LowerThresholdAlgorithm.get_name(),
                values=LowerThresholdAlgorithm.get_default_values(),
            ),
            mask_history=[
                SegmentationPipelineElement(
                    segmentation=ROIExtractionProfile(
                        name="test4",
                        algorithm=LowerThresholdAlgorithm.get_name(),
                        values=LowerThresholdAlgorithm.get_default_values(),
                    ),
                    mask_property=MaskProperty.simple_mask(),
                )
            ],
        )
        part_settings.roi_pipelines["test2"] = SegmentationPipeline(
            name="test2",
            segmentation=ROIExtractionProfile(
                name="test5",
                algorithm=LowerThresholdAlgorithm.get_name(),
                values=LowerThresholdAlgorithm.get_default_values(),
            ),
            mask_history=[
                SegmentationPipelineElement(
                    segmentation=ROIExtractionProfile(
                        name="test4",
                        algorithm=UpperThresholdAlgorithm.get_name(),
                        values=UpperThresholdAlgorithm.get_default_values(),
                    ),
                    mask_property=MaskProperty.simple_mask(),
                )
            ],
        )

        widget = prepare_plan_widget.ROIExtraction(part_settings)
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.segment_stack.currentChanged):
            widget.segment_stack.setCurrentWidget(widget.pipeline_profile)
        assert widget.pipeline_profile.count() == 2
        with qtbot.assert_not_emitted(widget.roi_extraction_pipeline_add):
            widget._add_profile()
        with qtbot.waitSignal(widget.roi_extraction_pipeline_selected, check_params_cb=check_pipeline("test")):
            widget.pipeline_profile.setCurrentRow(0)
        with qtbot.waitSignal(widget.roi_extraction_pipeline_selected, check_params_cb=check_pipeline("test2")):
            widget.pipeline_profile.setCurrentRow(1)

        with widget.enable_protect():
            with qtbot.assert_not_emitted(widget.roi_extraction_pipeline_selected):
                widget.pipeline_profile.setCurrentRow(0)
        with qtbot.waitSignal(widget.roi_extraction_pipeline_add, check_params_cb=check_pipeline("test")):
            widget._add_profile()

    def test_set_node_type(self, qtbot, part_settings):
        part_settings.roi_profiles["test"] = ROIExtractionProfile(
            name="test",
            algorithm=LowerThresholdAlgorithm.get_name(),
            values=LowerThresholdAlgorithm.get_default_values(),
        )
        widget = prepare_plan_widget.ROIExtraction(part_settings)
        qtbot.addWidget(widget)
        assert not widget.chose_profile_btn.isEnabled()
        widget.set_current_node(NodeType.root)
        assert not widget.chose_profile_btn.isEnabled()
        widget.set_current_node(None)
        with qtbot.waitSignal(widget.roi_extraction_profile_selected):
            widget.roi_extraction_profile.setCurrentRow(0)
        assert not widget.chose_profile_btn.isEnabled()
        widget.set_current_node(NodeType.root)
        assert widget.chose_profile_btn.isEnabled()
        widget.set_current_node(NodeType.mask)
        assert widget.chose_profile_btn.isEnabled()
        widget.set_current_node(NodeType.file_mask)
        assert widget.chose_profile_btn.isEnabled()
        with qtbot.waitSignal(widget.segment_stack.currentChanged):
            widget.segment_stack.setCurrentWidget(widget.pipeline_profile)
        assert not widget.chose_profile_btn.isEnabled()

    def test_replace(self, qtbot, part_settings):
        widget = prepare_plan_widget.ROIExtraction(part_settings)
        qtbot.addWidget(widget)
        assert widget.chose_profile_btn.text() == "Add Profile"
        widget.set_replace(True)
        assert widget.chose_profile_btn.text() == "Replace Profile"
        widget.set_replace(False)
        assert widget.chose_profile_btn.text() == "Add Profile"
        with qtbot.waitSignal(widget.segment_stack.currentChanged):
            widget.segment_stack.setCurrentWidget(widget.pipeline_profile)
        assert widget.chose_profile_btn.text() == "Add Pipeline"
        widget.set_replace(True)
        assert widget.chose_profile_btn.text() == "Replace Pipeline"


class TestSetOfMeasurement:
    def test_create(self, qtbot, part_settings):
        widget = prepare_plan_widget.SetOfMeasurement(part_settings)
        qtbot.addWidget(widget)

    def test_replace(self, qtbot, part_settings):
        widget = prepare_plan_widget.SetOfMeasurement(part_settings)
        qtbot.addWidget(widget)
        widget.set_replace(True)
        assert widget.add_calculation_btn.text() == "Replace set of measurements"
        widget.set_replace(False)
        assert widget.add_calculation_btn.text() == "Add set of measurements"

    def test_selected_profile(self, qtbot, part_settings):
        def check_measurement(name):
            def _check_measurement(measurement):
                return measurement == part_settings.measurement_profiles[name]

            return _check_measurement

        widget = prepare_plan_widget.SetOfMeasurement(part_settings)
        qtbot.addWidget(widget)
        with qtbot.waitSignal(widget.set_of_measurement_selected, check_params_cb=check_measurement("statistic1")):
            widget.measurements_list.setCurrentRow(0)
        with qtbot.waitSignal(widget.set_of_measurement_selected, check_params_cb=check_measurement("statistic2")):
            widget.measurements_list.setCurrentRow(1)
        with widget.enable_protect():
            with qtbot.assert_not_emitted(widget.set_of_measurement_selected):
                widget.measurements_list.setCurrentRow(0)

    def test_set_current_node(self, qtbot, part_settings):
        widget = prepare_plan_widget.SetOfMeasurement(part_settings)
        qtbot.addWidget(widget)
        widget.set_current_node(NodeType.segment)
        assert not widget.add_calculation_btn.isEnabled()
        widget.measurements_list.setCurrentRow(0)
        widget.set_current_node(NodeType.root)
        assert not widget.add_calculation_btn.isEnabled()
        widget.set_current_node(None)
        assert not widget.add_calculation_btn.isEnabled()
        widget.set_current_node(NodeType.segment)
        assert widget.add_calculation_btn.isEnabled()
        widget.set_current_node(NodeType.mask)
        assert not widget.add_calculation_btn.isEnabled()

    def test_add_measurement(self, qtbot, part_settings):
        def check_measurement(measurement: prepare_plan_widget.MeasurementCalculate):
            mes = copy(part_settings.measurement_profiles["statistic1"])
            mes.name_prefix = "prefix_"
            assert measurement.measurement_profile == mes
            assert measurement.channel == 4
            assert measurement.name_prefix == "prefix_"
            return True

        widget = prepare_plan_widget.SetOfMeasurement(part_settings)
        qtbot.addWidget(widget)
        assert widget.measurements_list.count() == 2
        with qtbot.assert_not_emitted(widget.set_of_measurement_add):
            widget._add_measurement()
        widget.measurements_list.setCurrentRow(0)
        widget.measurement_name_prefix.setText("prefix_")
        widget.choose_channel_for_measurements.setCurrentIndex(5)
        with qtbot.waitSignal(widget.set_of_measurement_add, check_params_cb=check_measurement):
            widget._add_measurement()


class TestCreatePlan:
    def test_create(self, qtbot, part_settings):
        widget = prepare_plan_widget.CreatePlan(part_settings)
        qtbot.addWidget(widget)
