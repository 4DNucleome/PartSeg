import os
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import pytest

from PartSegCore.algorithm_describe_base import SegmentationProfile
from PartSegCore.analysis.batch_processing import batch_backend
from PartSegCore.analysis.batch_processing.batch_backend import CalculationManager, CalculationProcess
from PartSegCore.analysis.calculation_plan import (
    Calculation,
    CalculationPlan,
    CalculationTree,
    FileCalculation,
    MaskCreate,
    MaskSuffix,
    MeasurementCalculate,
    RootType,
)
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, Node, PerComponent
from PartSegCore.analysis.measurement_calculation import MeasurementProfile
from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import Image, ImageWriter, TiffImageReader


class MocksCalculation:
    def __init__(self, file_path):
        self.file_path = file_path


@pytest.fixture
def create_test_data(tmpdir):
    # for future use
    spacing = tuple([x / UNIT_SCALE[Units.nm.value] for x in (210, 70, 70)])
    res = []
    for i in range(8):
        mask_data = np.zeros((10, 20, 20 + i), dtype=np.uint8)
        mask_data[1:-1, 2:-2, 2:-2] = 1
        data = np.zeros(mask_data.shape + (2,), dtype=np.uint16)
        data[1:-1, 2:-2, 2:-2] = 15000
        data[2:-2, 3:-3, 3:7] = 33000
        data[2:-2, 3:-3, -7:-3] = 33000
        image = Image(data, spacing, "", mask=mask_data, axes_order="ZYXC")
        ImageWriter.save(image, os.path.join(str(tmpdir), f"file_{i}.tif"))
        res.append(os.path.join(str(tmpdir), f"file_{i}.tif"))
        ImageWriter.save_mask(image, os.path.join(str(tmpdir), f"file_{i}_mask.tif"))
    return res


# TODO add check of per component measurements

# noinspection DuplicatedCode
class TestCalculationProcess:
    @staticmethod
    def create_calculation_plan():
        parameters = {
            "channel": 1,
            "minimum_size": 200,
            "threshold": {
                "name": "Base/Core",
                "values": {
                    "core_threshold": {"name": "Manual", "values": {"threshold": 30000}},
                    "base_threshold": {"name": "Manual", "values": {"threshold": 13000}},
                },
            },
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
            "sprawl_type": {"name": "Euclidean", "values": {}},
        }

        segmentation = SegmentationProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
        mask_suffix = MaskSuffix(name="", suffix="_mask")
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                name="Segmentation Volume/Mask Volume",
                calculation_tree=Node(
                    left=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
                    op="/",
                    right=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Segmentation Components Number",
                calculation_tree=Leaf("Components number", area=AreaType.ROI, per_component=PerComponent.No),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = MeasurementCalculate(
            channel=0, units=Units.µm, statistic_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            RootType.Image,
            [CalculationTree(mask_suffix, [CalculationTree(segmentation, [CalculationTree(statistic_calculate, [])])])],
        )
        return CalculationPlan(tree=tree, name="test")

    @staticmethod
    def create_calculation_plan2():
        parameters = {
            "channel": 0,
            "minimum_size": 200,
            "threshold": {
                "name": "Base/Core",
                "values": {
                    "core_threshold": {"name": "Manual", "values": {"threshold": 30000}},
                    "base_threshold": {"name": "Manual", "values": {"threshold": 13000}},
                },
            },
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
            "sprawl_type": {"name": "Euclidean", "values": {}},
        }

        segmentation = SegmentationProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                name="Segmentation Volume/Mask Volume",
                calculation_tree=Node(
                    left=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
                    op="/",
                    right=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Segmentation Components Number",
                calculation_tree=Leaf("Components number", area=AreaType.ROI, per_component=PerComponent.No),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = MeasurementCalculate(
            channel=0, units=Units.µm, statistic_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            RootType.Mask_project, [CalculationTree(segmentation, [CalculationTree(statistic_calculate, [])])]
        )
        return CalculationPlan(tree=tree, name="test2")

    @staticmethod
    def create_calculation_plan3():
        parameters = {
            "channel": 1,
            "minimum_size": 200,
            "threshold": {
                "name": "Base/Core",
                "values": {
                    "core_threshold": {"name": "Manual", "values": {"threshold": 30000}},
                    "base_threshold": {"name": "Manual", "values": {"threshold": 13000}},
                },
            },
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
            "sprawl_type": {"name": "Euclidean", "values": {}},
        }

        segmentation = SegmentationProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
        mask_suffix = MaskSuffix(name="", suffix="_mask")
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                name="Segmentation Volume/Mask Volume",
                calculation_tree=Node(
                    left=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
                    op="/",
                    right=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Segmentation Components Number",
                calculation_tree=Leaf("Components number", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Segmentation Volume per component",
                calculation_tree=Leaf("Volume", area=AreaType.ROI, per_component=PerComponent.Yes),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = MeasurementCalculate(
            channel=0, units=Units.µm, statistic_profile=statistic, name_prefix=""
        )
        mask_create = MaskCreate("", MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, True, False, False))
        parameters2 = {
            "channel": 1,
            "minimum_size": 200,
            "threshold": {"name": "Manual", "values": {"threshold": 30000}},
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
        }

        segmentation2 = SegmentationProfile(name="test", algorithm="Lower threshold", values=parameters2)
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                name="Segmentation Volume/Mask Volume",
                calculation_tree=Node(
                    left=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
                    op="/",
                    right=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.No),
                ),
            ),
            MeasurementEntry(
                "Segmentation Components Number",
                calculation_tree=Leaf("Components number", area=AreaType.ROI, per_component=PerComponent.No),
            ),
            MeasurementEntry(
                "Mask Volume per component",
                calculation_tree=Leaf("Volume", area=AreaType.Mask, per_component=PerComponent.Yes),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure2", chosen_fields=chosen_fields[:], name_prefix="aa_")
        statistic_calculate2 = MeasurementCalculate(
            channel=0, units=Units.µm, statistic_profile=statistic, name_prefix=""
        )
        chosen_fields.append(
            MeasurementEntry(
                "Segmentation Volume per component",
                calculation_tree=Leaf("Volume", area=AreaType.ROI, per_component=PerComponent.Yes),
            )
        )
        statistic = MeasurementProfile(name="base_measure3", chosen_fields=chosen_fields[:], name_prefix="bb_")
        statistic_calculate3 = MeasurementCalculate(
            channel=0, units=Units.µm, statistic_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            RootType.Image,
            [
                CalculationTree(
                    mask_suffix,
                    [
                        CalculationTree(
                            segmentation,
                            [
                                CalculationTree(statistic_calculate, []),
                                CalculationTree(
                                    mask_create,
                                    [
                                        CalculationTree(
                                            segmentation2,
                                            [
                                                CalculationTree(statistic_calculate2, []),
                                                CalculationTree(statistic_calculate3, []),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        )
                    ],
                )
            ],
        )
        return CalculationPlan(tree=tree, name="test")

    def test_one_file(self, data_test_dir):
        plan = self.create_calculation_plan()
        process = CalculationProcess()
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component5.tif")
        calc = MocksCalculation(file_path)
        process.calculation = calc
        process.image = TiffImageReader.read_image(file_path)
        process.iterate_over(plan.execution_tree)
        assert len(process.measurement[0]) == 3

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline(self, tmpdir, data_test_dir, monkeypatch):
        monkeypatch.setattr(batch_backend, "CalculationProcess", MockCalculationProcess)
        plan = self.create_calculation_plan()
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(3)
        manager.add_calculation(calc)

        while manager.has_work:
            time.sleep(0.1)
            manager.get_results()
        manager.writer.finish()
        if sys.platform == "darwin":
            time.sleep(2)
        else:
            time.sleep(0.4)
        assert os.path.exists(os.path.join(tmpdir, "test.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test.xlsx"), index_col=0, header=[0, 1])
        assert df.shape == (8, 4)
        for i in range(8):
            assert os.path.basename(df.name.units[i]) == f"stack1_component{i+1}.tif"

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_mask_project(self, tmpdir, data_test_dir):
        plan = self.create_calculation_plan2()
        file_pattern = os.path.join(data_test_dir, "*nucleus.seg")
        file_paths = glob(file_pattern)
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test2.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(2)
        manager.add_calculation(calc)

        while manager.has_work:
            time.sleep(0.1)
            manager.get_results()
        if sys.platform == "darwin":
            time.sleep(2)
        else:
            time.sleep(0.4)
        manager.writer.finish()
        assert os.path.exists(os.path.join(tmpdir, "test2.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test2.xlsx"), index_col=0, header=[0, 1])
        assert df.shape == (2, 4)

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_component_split(self, tmpdir, data_test_dir):
        plan = self.create_calculation_plan3()
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = glob(file_pattern)
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(2)
        manager.add_calculation(calc)

        while manager.has_work:
            time.sleep(0.1)
            res = manager.get_results()
            if res.errors:
                print(res.errors, file=sys.stderr)
        if sys.platform == "darwin":
            time.sleep(2)
        else:
            time.sleep(0.4)
        manager.writer.finish()
        assert os.path.exists(os.path.join(tmpdir, "test3.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), index_col=0, header=[0, 1])
        assert df.shape == (8, 10)
        df2 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=1, index_col=0, header=[0, 1])
        assert df2.shape[0] > 8
        assert df2.shape == (df["Segmentation Components Number"]["count"].sum(), 6)
        df3 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=2, index_col=0, header=[0, 1])
        assert df3.shape == (df["Segmentation Components Number"]["count"].sum(), 6)
        df4 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=3, index_col=0, header=[0, 1])
        assert df4.shape == (df["Segmentation Components Number"]["count"].sum(), 8)


class MockCalculationProcess(CalculationProcess):
    def do_calculation(self, calculation: FileCalculation):
        if os.path.basename(calculation.file_path) == "stack1_component1.tif":
            time.sleep(0.5)
        return super().do_calculation(calculation)
