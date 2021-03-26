import os
import shutil
import sys
import time
from glob import glob

import numpy as np
import pandas as pd
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.batch_processing import batch_backend
from PartSegCore.analysis.batch_processing.batch_backend import (
    CalculationManager,
    CalculationProcess,
    ResponseData,
    do_calculation,
)
from PartSegCore.analysis.calculation_plan import (
    Calculation,
    CalculationPlan,
    CalculationTree,
    FileCalculation,
    MaskCreate,
    MaskIntersection,
    MaskSuffix,
    MaskSum,
    MaskUse,
    MeasurementCalculate,
    RootType,
    Save,
)
from PartSegCore.analysis.measurement_base import AreaType, Leaf, MeasurementEntry, Node, PerComponent
from PartSegCore.analysis.measurement_calculation import MeasurementProfile
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore.image_operations import RadiusType
from PartSegCore.io_utils import SaveBase
from PartSegCore.mask_create import MaskProperty
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.universal_const import UNIT_SCALE, Units
from PartSegImage import Image, ImageWriter, TiffImageReader

ENGINE = None if pd.__version__ == "0.24.0" else "openpyxl"


class MocksCalculation:
    def __init__(self, file_path):
        self.file_path = file_path


@pytest.fixture
def create_test_data(tmpdir):
    # for future use
    spacing = tuple(x / UNIT_SCALE[Units.nm.value] for x in (210, 70, 70))
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

        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
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
            channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
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

        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
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
            channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            RootType.Mask_project, [CalculationTree(segmentation, [CalculationTree(statistic_calculate, [])])]
        )
        return CalculationPlan(tree=tree, name="test2")

    @staticmethod
    def create_simple_plan(root_type: RootType, save: Save):
        parameters = {
            "channel": 0,
            "minimum_size": 200,
            "threshold": {"name": "Manual", "values": {"threshold": 13000}},
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
        }
        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold", values=parameters)
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = MeasurementCalculate(
            channel=-1, units=Units.µm, measurement_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            root_type,
            [CalculationTree(segmentation, [CalculationTree(statistic_calculate, []), CalculationTree(save, [])])],
        )
        return CalculationPlan(tree=tree, name="test")

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

        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)
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
            channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
        )
        mask_create = MaskCreate("", MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, True, False, False))
        parameters2 = {
            "channel": 1,
            "minimum_size": 200,
            "threshold": {"name": "Manual", "values": {"threshold": 30000}},
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
        }

        segmentation2 = ROIExtractionProfile(name="test", algorithm="Lower threshold", values=parameters2)
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
            channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
        )
        chosen_fields.append(
            MeasurementEntry(
                "Segmentation Volume per component",
                calculation_tree=Leaf("Volume", area=AreaType.ROI, per_component=PerComponent.Yes),
            )
        )
        statistic = MeasurementProfile(name="base_measure3", chosen_fields=chosen_fields[:], name_prefix="bb_")
        statistic_calculate3 = MeasurementCalculate(
            channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
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

    @staticmethod
    def create_mask_operation_plan(mask_op):
        parameters = {
            "channel": 0,
            "minimum_size": 200,
            "threshold": {"name": "Manual", "values": {"threshold": 13000}},
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
        }
        parameters2 = dict(**parameters)
        parameters2["channel"] = 1
        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold", values=parameters)
        segmentation2 = ROIExtractionProfile(name="test2", algorithm="Lower threshold", values=parameters2)
        chosen_fields = [
            MeasurementEntry(
                name="Segmentation Volume",
                calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
            ),
        ]
        statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
        statistic_calculate = MeasurementCalculate(
            channel=-1, units=Units.µm, measurement_profile=statistic, name_prefix=""
        )
        tree = CalculationTree(
            RootType.Image,
            [
                CalculationTree(
                    segmentation,
                    [CalculationTree(MaskCreate(name="test1", mask_property=MaskProperty.simple_mask()), [])],
                ),
                CalculationTree(
                    segmentation2,
                    [CalculationTree(MaskCreate(name="test2", mask_property=MaskProperty.simple_mask()), [])],
                ),
                CalculationTree(mask_op, [CalculationTree(segmentation2, [CalculationTree(statistic_calculate, [])])]),
            ],
        )
        return CalculationPlan(tree=tree, name="test")

    @pytest.mark.parametrize(
        "mask_op", [MaskUse("test1"), MaskSum("", "test1", "test2"), MaskIntersection("", "test1", "test2")]
    )
    def test_mask_op(self, mask_op, data_test_dir, tmpdir):
        plan = self.create_mask_operation_plan(mask_op)
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=os.path.dirname(file_path),
            result_prefix=tmpdir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

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
    def test_full_pipeline_base(self, tmpdir, data_test_dir, monkeypatch):
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
        calc_process = CalculationProcess()
        for file_path in file_paths:
            res = calc_process.do_calculation(FileCalculation(file_path, calc))
            assert isinstance(res, list)
            assert isinstance(res[0], ResponseData)

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

        for _ in range(int(120 / 0.1)):
            manager.get_results()
            if manager.has_work:
                time.sleep(0.1)
            else:
                break
        else:
            manager.kill_jobs()
            pytest.fail("jobs hanged")

        manager.writer.finish()
        if sys.platform == "darwin":
            time.sleep(2)
        else:
            time.sleep(0.4)
        assert os.path.exists(os.path.join(tmpdir, "test.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (8, 4)
        for i in range(8):
            assert os.path.basename(df.name.units[i]) == f"stack1_component{i+1}.tif"

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_error(self, tmp_path_factory, data_test_dir, monkeypatch):
        plan = self.create_calculation_plan()
        data_dir = tmp_path_factory.mktemp("data")
        file_pattern_copy = os.path.join(data_test_dir, "stack1_components", "stack1_component*.tif")
        file_paths = sorted(glob(file_pattern_copy))
        for el in file_paths:
            shutil.copy(el, data_dir)
            shutil.copy(data_dir / "stack1_component1.tif", data_dir / "stack1_component10.tif")
        file_pattern = os.path.join(data_dir, "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        result_dir = tmp_path_factory.mktemp("result")

        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=str(data_dir),
            result_prefix=str(data_dir),
            measurement_file_path=os.path.join(result_dir, "test.xlsx"),
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
        assert os.path.exists(os.path.join(result_dir, "test.xlsx"))
        df = pd.read_excel(os.path.join(result_dir, "test.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (8, 4)
        for i in range(8):
            assert os.path.basename(df.name.units[i]) == f"stack1_component{i + 1}.tif"
        df2 = pd.read_excel(os.path.join(result_dir, "test.xlsx"), sheet_name="Errors", index_col=0, engine=ENGINE)
        assert df2.shape == (1, 2)
        str(df2.loc[0]["error description"]).startswith("[Errno 2]")

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
        df = pd.read_excel(os.path.join(tmpdir, "test2.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (2, 4)

    def test_do_calculation(self, tmpdir, data_test_dir):
        plan = self.create_calculation_plan3()
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )
        index, res = do_calculation((1, file_path), calc)
        assert index == 1
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

    @pytest.mark.parametrize(
        "file_name,root_type",
        [
            (os.path.join("stack1_components", "stack1_component1.tif"), RootType.Image),
            ("stack1_component1_1.tgz", RootType.Image),
            ("stack1_component1_1.tgz", RootType.Project),
            ("test_nucleus_1_1.seg", RootType.Image),
            ("test_nucleus_1_1.seg", RootType.Mask_project),
        ],
    )
    @pytest.mark.parametrize("save_method", save_dict.values())
    def test_do_calculation_save(self, tmpdir, data_test_dir, file_name, root_type, save_method: SaveBase):
        save_desc = Save(
            "_test", "", save_method.get_name(), save_method.get_short_name(), save_method.get_default_values()
        )
        plan = self.create_simple_plan(root_type, save_desc)
        file_path = os.path.join(data_test_dir, file_name)
        calc = Calculation(
            [file_path],
            base_prefix=os.path.dirname(file_path),
            result_prefix=tmpdir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

    def test_do_calculation_calculation_process(self, tmpdir, data_test_dir):
        plan = self.create_calculation_plan3()
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=plan,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

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
        df = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (8, 10)
        df2 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=1, index_col=0, header=[0, 1], engine=ENGINE)
        assert df2.shape[0] > 8
        assert df2.shape == (df["Segmentation Components Number"]["count"].sum(), 6)
        df3 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=2, index_col=0, header=[0, 1], engine=ENGINE)
        assert df3.shape == (df["Segmentation Components Number"]["count"].sum(), 6)
        df4 = pd.read_excel(os.path.join(tmpdir, "test3.xlsx"), sheet_name=3, index_col=0, header=[0, 1], engine=ENGINE)
        assert df4.shape == (df["Segmentation Components Number"]["count"].sum(), 8)


class MockCalculationProcess(CalculationProcess):
    def do_calculation(self, calculation: FileCalculation):
        if os.path.basename(calculation.file_path) == "stack1_component1.tif":
            time.sleep(0.5)
        return super().do_calculation(calculation)
