import pytest

from PartSeg._roi_analysis.batch_window import CalculationPrepare
from PartSegCore.analysis.batch_processing.batch_backend import CalculationManager
from PartSegCore.analysis.calculation_plan import CalculationPlan, CalculationTree, MaskSuffix, RootType


@pytest.fixture
def calculation_prepare(tmp_path, part_settings, qtbot):
    def _constructor(file_list, calculation_plan=None):
        if calculation_plan is None:
            calculation_plan = CalculationPlan()
        dial = CalculationPrepare(
            file_list=file_list,
            calculation_plan=calculation_plan,
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        return dial

    return _constructor


class TestCalculationPrepare:
    @pytest.mark.parametrize("files_li", [["test1.tif", "test2.tif"], ["test1.tif"]])
    def test_init(self, calculation_prepare, tmp_path, files_li):
        files = [tmp_path / x for x in files_li]
        for file in files:
            file.write_text("test")
        dial = calculation_prepare(file_list=files)
        assert dial.all_file_prefix == str(tmp_path)
        assert dial.execute_btn.isEnabled()

    def test_show(self, calculation_prepare, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif", tmp_path / "test2.aaa"]
        for file in files:
            file.write_text("test")
        dial = calculation_prepare(file_list=files)
        assert dial.file_list_widget.topLevelItemCount() == 0
        dial._show_event_setup()
        assert dial.file_list_widget.topLevelItemCount() == 3

    def test_no_file(self, calculation_prepare, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")

        files.append(tmp_path / "test3.tif")

        dial = calculation_prepare(file_list=files)
        assert not dial.execute_btn.isEnabled()

    def test_no_mask(self, calculation_prepare, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
            file.with_name(file.with_suffix("").name + "_mask.tif").write_text("test")

        files.append(tmp_path / "test3.tif")
        files[-1].write_text("test")

        dial = calculation_prepare(
            file_list=files,
            calculation_plan=CalculationPlan(
                CalculationTree(RootType.Image, [CalculationTree(MaskSuffix(name="", suffix="_mask"), [])])
            ),
        )
        dial._check_start_conditions()
        assert not dial.execute_btn.isEnabled()

    def test_choose_data_prefix(self, calculation_prepare, tmp_path, monkeypatch):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        monkeypatch.setattr("PartSeg._roi_analysis.batch_window.QFileDialog.exec_", lambda *_: True)
        monkeypatch.setattr(
            "PartSeg._roi_analysis.batch_window.QFileDialog.selectedFiles", lambda *_: [tmp_path / "test_dir"]
        )
        dial = calculation_prepare(file_list=files)
        dial.choose_data_prefix()

        assert dial.base_prefix.text() == str(tmp_path / "test_dir")

    def test_choose_result_prefix(self, calculation_prepare, tmp_path, monkeypatch):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        monkeypatch.setattr("PartSeg._roi_analysis.batch_window.QFileDialog.exec_", lambda *_: True)
        monkeypatch.setattr(
            "PartSeg._roi_analysis.batch_window.QFileDialog.selectedFiles", lambda *_: [tmp_path / "test_dir"]
        )
        dial = calculation_prepare(file_list=files)
        dial.choose_result_prefix()

        assert dial.result_prefix.text() == str(tmp_path / "test_dir")

    def test_overwrite_voxel_size(self, calculation_prepare, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        dial = calculation_prepare(file_list=files)
        assert "ignored" not in dial.info_label.text()
        dial.overwrite_voxel_size_check.setChecked(True)
        assert "ignored" in dial.info_label.text()
        dial.overwrite_voxel_size_check.setChecked(False)
        assert "ignored" not in dial.info_label.text()
