import os
from glob import glob
import time
import pandas as pd

from PartSeg.tiff_image import ImageReader
from PartSeg.utils.analysis.algorithm_description import SegmentationProfile
from PartSeg.utils.analysis.batch_processing.batch_backend import CalculationProcess, CalculationManager
from PartSeg.utils.analysis.calculation_plan import CalculationPlan, CalculationTree, MaskSuffix, StatisticCalculate, \
    Calculation
from PartSeg.utils.analysis.statistics_calculation import StatisticProfile, StatisticEntry, Leaf, AreaType, \
    PerComponent, Node
from PartSeg.utils.segmentation.noise_filtering import GaussType
from PartSeg.utils.universal_const import Units

from help_fun import get_test_dir


class MocksCalculation:
    def __init__(self, file_path):
        self.file_path = file_path


class TestCalculationProcess:
    @staticmethod
    def create_calculation_plan():
        parameters = {"channel": 1, "minimum_size": 200,
                      'threshold': {'name': 'Double Choose',
                                    'values': {
                                        'core_threshold': {'name': 'Manual', 'values': {'threshold': 30000}},
                                        'base_threshold': {'name': 'Manual', 'values': {'threshold': 13000}}}},
                      'noise_removal': {'name': 'Gauss', 'values': {'gauss_type': GaussType.Layer, "radius": 1.0}},
                      'side_connection': False,
                      'sprawl_type': {'name': 'Euclidean sprawl', 'values': {}}}

        segmentation = SegmentationProfile(name="test", algorithm="Lower threshold flow", values=parameters)
        mask_suffix = MaskSuffix(name="", suffix="_mask")
        chosen_fields = [StatisticEntry(name="Segmentation Volume", calculation_tree=Leaf(
            name="Volume", area=AreaType.Segmentation, per_component=PerComponent.No)),
                         StatisticEntry(
                             name="Segmentation Volume/Mask Volume",
                             calculation_tree=Node(
                                 left=Leaf(name="Volume", area=AreaType.Segmentation, per_component=PerComponent.No),
                                 op="/",
                                 right=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.No))),
                         StatisticEntry("Segmentation Components Number",
                                        calculation_tree=Leaf("Components Number", area=AreaType.Segmentation,
                                                              per_component=PerComponent.No))]
        statistic = StatisticProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = StatisticCalculate(channel=0, units=Units.µm, statistic_profile=statistic, name_prefix="")
        tree = CalculationTree("root",
                               [CalculationTree(mask_suffix,
                                                [CalculationTree(segmentation,
                                                                 [CalculationTree(statistic_calculate, [])])])])
        return CalculationPlan(tree=tree, name="test")

    def test_one_file(self):
        plan = self.create_calculation_plan()
        process = CalculationProcess()
        file_path = os.path.join(get_test_dir(), "stack1_components", "stack1_component5.tif")
        calc = MocksCalculation(file_path)
        process.calculation = calc
        process.image = ImageReader.read_image(file_path)
        process.iterate_over(plan.execution_tree)
        print(process.statistics)
        assert (len(process.statistics[0]) == 3)

    def test_full_pipeline(self):
        plan = self.create_calculation_plan()
        file_pattern = os.path.join(get_test_dir(), "stack1_components", "stack1_component*[0-9].tif")
        file_paths = glob(file_pattern)
        calc = Calculation(file_paths, base_prefix=get_test_dir(), result_prefix=get_test_dir(),
                           statistic_file_path=os.path.join(get_test_dir(), "test.xlsx"), sheet_name="Sheet1",
                           calculation_plan=plan, voxel_size=(1, 1, 1))

        manager = CalculationManager()
        manager.set_number_of_workers(2)
        manager.add_calculation(calc)

        while manager.has_work:
            time.sleep(0.1)
            __ = manager.get_results()
        time.sleep(0.4)
        assert os.path.exists(os.path.join(get_test_dir(), "test.xlsx"))
        df = pd.read_excel(os.path.join(get_test_dir(), "test.xlsx"), index_col=0, header=[0, 1])
        assert df.shape == (8, 4)
