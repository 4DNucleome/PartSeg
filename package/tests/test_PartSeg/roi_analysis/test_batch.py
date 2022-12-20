from PartSeg._roi_analysis.batch_window import CalculationPrepare
from PartSegCore.analysis.batch_processing.batch_backend import CalculationManager
from PartSegCore.analysis.calculation_plan import CalculationPlan


class TestCalculationPrepare:
    def test_init(self, part_settings, qtbot, tmp_path):
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
