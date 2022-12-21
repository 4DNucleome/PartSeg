from PartSeg._roi_analysis.batch_window import CalculationPrepare
from PartSegCore.analysis.batch_processing.batch_backend import CalculationManager
from PartSegCore.analysis.calculation_plan import CalculationPlan, CalculationTree, MaskSuffix, RootType


class TestCalculationPrepare:
    def test_init(self, part_settings, qtbot, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        dial = CalculationPrepare(
            file_list=files,
            calculation_plan=CalculationPlan(),
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        assert dial.execute_btn.isEnabled()

    def test_no_file(self, part_settings, qtbot, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")

        files.append(tmp_path / "test3.tif")

        dial = CalculationPrepare(
            file_list=files,
            calculation_plan=CalculationPlan(),
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        assert not dial.execute_btn.isEnabled()

    def test_no_mask(self, part_settings, qtbot, tmp_path):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
            file.with_name(file.with_suffix("").name + "_mask.tif").write_text("test")

        files.append(tmp_path / "test3.tif")
        files[-1].write_text("test")

        dial = CalculationPrepare(
            file_list=files,
            calculation_plan=CalculationPlan(
                CalculationTree(RootType.Image, [CalculationTree(MaskSuffix(name="", suffix="_mask"), [])])
            ),
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        dial._check_start_conditions()
        assert not dial.execute_btn.isEnabled()

    def test_choose_data_prefix(self, part_settings, qtbot, tmp_path, monkeypatch):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        monkeypatch.setattr("PartSeg._roi_analysis.batch_window.QFileDialog.exec_", lambda *_: True)
        monkeypatch.setattr(
            "PartSeg._roi_analysis.batch_window.QFileDialog.selectedFiles", lambda *_: [tmp_path / "test_dir"]
        )
        dial = CalculationPrepare(
            file_list=files,
            calculation_plan=CalculationPlan(),
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        dial.choose_data_prefix()

        assert dial.base_prefix.text() == str(tmp_path / "test_dir")

    def test_choose_result_prefix(self, part_settings, qtbot, tmp_path, monkeypatch):
        files = [tmp_path / "test1.tif", tmp_path / "test2.tif"]
        for file in files:
            file.write_text("test")
        monkeypatch.setattr("PartSeg._roi_analysis.batch_window.QFileDialog.exec_", lambda *_: True)
        monkeypatch.setattr(
            "PartSeg._roi_analysis.batch_window.QFileDialog.selectedFiles", lambda *_: [tmp_path / "test_dir"]
        )
        dial = CalculationPrepare(
            file_list=files,
            calculation_plan=CalculationPlan(),
            measurement_file_path=tmp_path / "test1.xlsx",
            settings=part_settings,
            batch_manager=CalculationManager(),
        )
        qtbot.addWidget(dial)
        dial.choose_result_prefix()

        assert dial.result_prefix.text() == str(tmp_path / "test_dir")
