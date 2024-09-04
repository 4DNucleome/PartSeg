# pylint: disable=no-self-use
import json
import os
import shutil
import sys
import time
from glob import glob
from itertools import dropwhile
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import tifffile

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import AnalysisAlgorithmSelection
from PartSegCore.analysis.batch_processing import batch_backend
from PartSegCore.analysis.batch_processing.batch_backend import (
    CalculationManager,
    CalculationProcess,
    ResponseData,
    SheetData,
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
from PartSegCore.io_utils import LoadPlanExcel, SaveBase
from PartSegCore.json_hooks import PartSegEncoder
from PartSegCore.mask.io_functions import MaskProjectTuple, SaveROI, SaveROIOptions
from PartSegCore.mask_create import MaskProperty
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation import ROIExtractionAlgorithm, ROIExtractionResult
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.segmentation.restartable_segmentation_algorithms import LowerThresholdFlowAlgorithm
from PartSegCore.universal_const import Units
from PartSegCore.utils import BaseModel
from PartSegImage import Channel, Image, ImageWriter, TiffImageReader

ENGINE = None if pd.__version__ == "0.24.0" else "openpyxl"


class MocksCalculation:
    def __init__(self, file_path):
        self.file_path = file_path


# TODO add check of per component measurements


class DummyParams(BaseModel):
    channel: Channel = 0


class DummyExtraction(ROIExtractionAlgorithm):
    __argument_class__ = DummyParams

    @classmethod
    def support_time(cls):
        return True

    @classmethod
    def support_z(cls):  # pragma: no cover
        return True

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        channel = self.image.get_channel(0)
        if channel.max() == 0:
            raise ValueError("Empty image")
        return ROIExtractionResult(np.ones(channel.shape, dtype=np.uint8), self.get_segmentation_profile())

    def get_info_text(self):  # pragma: no cover
        return ""

    @classmethod
    def get_name(cls) -> str:
        return "Dummy"


class DummySpacingCheck(DummyExtraction):
    def calculation_run(self, report_fun: Callable[[str, int], None]) -> ROIExtractionResult:
        assert self.image.spacing == (3, 2, 1)
        return ROIExtractionResult(np.ones(self.image.shape, dtype=np.uint8), self.get_segmentation_profile())


@pytest.fixture
def _register_dummy_extraction():
    assert "Dummy" not in AnalysisAlgorithmSelection.__register__
    AnalysisAlgorithmSelection.register(DummyExtraction)
    yield
    AnalysisAlgorithmSelection.__register__.pop("Dummy")
    assert "Dummy" not in AnalysisAlgorithmSelection.__register__


@pytest.fixture
def _register_dummy_spacing():
    assert "Dummy" not in AnalysisAlgorithmSelection.__register__
    AnalysisAlgorithmSelection.register(DummySpacingCheck)
    yield
    AnalysisAlgorithmSelection.__register__.pop("Dummy")
    assert "Dummy" not in AnalysisAlgorithmSelection.__register__


@pytest.fixture
def _prepare_spacing_data(tmp_path):
    data = np.zeros((4, 1, 10, 10), dtype=np.uint8)
    data[:, :, 2:-2, 2:-2] = 1
    tifffile.imwrite(tmp_path / "test1.tiff", data)

    image = Image(data, (1, 1, 1), axes_order="ZCYX", file_path=tmp_path / "test2.tiff")
    ImageWriter.save(image, image.file_path)


@pytest.fixture
def _prepare_mask_project_data(tmp_path):
    data = np.zeros((4, 10, 10), dtype=np.uint8)
    data[:, 2:4, 2:4] = 1
    data[:, 6:8, 2:4] = 2
    data[:, 6:8, 6:8] = 3

    image = Image(data, (1, 1, 1), axes_order="ZYX", file_path=tmp_path / "test.tiff")
    ImageWriter.save(image, image.file_path)

    roi = np.zeros(data.shape, dtype=np.uint8)
    roi[:, :5, :5] = 1
    roi[:, 5:10, :5] = 2
    roi[:, :5, 5:10] = 3
    roi[:, 5:10, 5:10] = 4

    roi = image.fit_mask_to_image(roi)

    proj = MaskProjectTuple(file_path=image.file_path, image=image, roi_info=ROIInfo(roi))

    SaveROI.save(tmp_path / "test.seg", proj, SaveROIOptions())


@pytest.fixture
def ltww_segmentation():
    parameters = LowerThresholdFlowAlgorithm.__argument_class__(
        channel=1,
        minimum_size=200,
        threshold={
            "name": "Base/Core",
            "values": {
                "core_threshold": {"name": "Manual", "values": {"threshold": 30000}},
                "base_threshold": {"name": "Manual", "values": {"threshold": 13000}},
            },
        },
        noise_filtering={"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
        side_connection=False,
        flow_type={"name": "Euclidean", "values": {}},
    )

    return ROIExtractionProfile(name="test", algorithm="Lower threshold with watershed", values=parameters)


@pytest.fixture
def measurement_list():
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
            name="Segmentation Components Number",
            calculation_tree=Leaf(name="Components number", area=AreaType.ROI, per_component=PerComponent.No),
        ),
    ]
    statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
    return MeasurementCalculate(channel=0, units=Units.µm, measurement_profile=statistic, name_prefix="")


@pytest.fixture
def simple_measurement_list():
    chosen_fields = [
        MeasurementEntry(
            name="Segmentation Volume",
            calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.No),
        ),
    ]
    statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
    return MeasurementCalculate(channel=-1, units=Units.µm, measurement_profile=statistic, name_prefix="")


@pytest.fixture
def calculation_plan_dummy(simple_measurement_list):
    tree = CalculationTree(
        RootType.Mask_project,
        [
            CalculationTree(
                ROIExtractionProfile(name="test", algorithm=DummyExtraction.get_name(), values=DummyParams()),
                [CalculationTree(simple_measurement_list, [])],
            )
        ],
    )
    return CalculationPlan(tree=tree, name="test")


@pytest.fixture
def calculation_plan_dummy_spacing(calculation_plan_dummy):
    calculation_plan_dummy.execution_tree.operation = RootType.Image
    return calculation_plan_dummy


@pytest.fixture
def calculation_plan(ltww_segmentation, measurement_list):
    mask_suffix = MaskSuffix(name="", suffix="_mask")
    tree = CalculationTree(
        operation=RootType.Image,
        children=[
            CalculationTree(mask_suffix, [CalculationTree(ltww_segmentation, [CalculationTree(measurement_list, [])])])
        ],
    )
    return CalculationPlan(tree=tree, name="test")


@pytest.fixture
def calculation_plan_long(ltww_segmentation, measurement_list):
    mask_suffix = MaskSuffix(name="", suffix="_mask")
    children = []
    for i in range(20):
        measurement = measurement_list.copy()
        measurement.name_prefix = f"{i}_"
        measurement.measurement_profile = measurement.measurement_profile.copy()
        measurement.measurement_profile.name_prefix = f"{i}_"
        children.append(
            CalculationTree(mask_suffix, [CalculationTree(ltww_segmentation, [CalculationTree(measurement, [])])])
        )
    tree = CalculationTree(
        operation=RootType.Image,
        children=children,
    )
    return CalculationPlan(tree=tree, name="test")


@pytest.fixture
def calculation_plan2(ltww_segmentation, measurement_list):
    ltww_segmentation.values.channel = 0

    tree = CalculationTree(
        RootType.Mask_project, [CalculationTree(ltww_segmentation, [CalculationTree(measurement_list, [])])]
    )
    return CalculationPlan(tree=tree, name="test2")


@pytest.fixture
def simple_plan(simple_measurement_list):
    def _create_simple_plan(root_type: RootType, save: Save):
        parameters = {
            "channel": 0,
            "minimum_size": 200,
            "threshold": {"name": "Manual", "values": {"threshold": 13000}},
            "noise_filtering": {"name": "Gauss", "values": {"dimension_type": DimensionType.Layer, "radius": 1.0}},
            "side_connection": False,
        }
        segmentation = ROIExtractionProfile(name="test", algorithm="Lower threshold", values=parameters)
        tree = CalculationTree(
            root_type,
            [CalculationTree(segmentation, [CalculationTree(simple_measurement_list, []), CalculationTree(save, [])])],
        )
        return CalculationPlan(tree=tree, name="test")

    return _create_simple_plan


@pytest.fixture
def calculation_plan3(ltww_segmentation):
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
            name="Segmentation Components Number",
            calculation_tree=Leaf(name="Components number", area=AreaType.ROI, per_component=PerComponent.No),
        ),
        MeasurementEntry(
            name="Segmentation Volume per component",
            calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.Yes),
        ),
    ]
    statistic = MeasurementProfile(name="base_measure", chosen_fields=chosen_fields, name_prefix="")
    statistic_calculate = MeasurementCalculate(channel=0, units=Units.µm, measurement_profile=statistic, name_prefix="")
    mask_create = MaskCreate(
        name="",
        mask_property=MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.NO,
            max_holes_size=0,
            save_components=True,
            clip_to_mask=False,
            reversed_mask=False,
        ),
    )
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
            name="Segmentation Components Number",
            calculation_tree=Leaf(name="Components number", area=AreaType.ROI, per_component=PerComponent.No),
        ),
        MeasurementEntry(
            name="Mask Volume per component",
            calculation_tree=Leaf(name="Volume", area=AreaType.Mask, per_component=PerComponent.Yes),
        ),
    ]
    statistic = MeasurementProfile(name="base_measure2", chosen_fields=chosen_fields[:], name_prefix="aa_")
    statistic_calculate2 = MeasurementCalculate(
        channel=0, units=Units.µm, measurement_profile=statistic, name_prefix=""
    )
    chosen_fields.append(
        MeasurementEntry(
            name="Segmentation Volume per component",
            calculation_tree=Leaf(name="Volume", area=AreaType.ROI, per_component=PerComponent.Yes),
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
                        ltww_segmentation,
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


@pytest.fixture(
    params=[
        MaskUse(name="test1"),
        MaskSum(name="", mask1="test1", mask2="test2"),
        MaskIntersection(name="", mask1="test1", mask2="test2"),
    ]
)
def mask_operation_plan(request, simple_measurement_list):
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
            CalculationTree(
                request.param, [CalculationTree(segmentation2, [CalculationTree(simple_measurement_list, [])])]
            ),
        ],
    )
    return CalculationPlan(tree=tree, name="test")


def wait_for_calculation(manager):
    for _ in range(int(120 / 0.1)):
        res = manager.get_results()
        if res.errors and res.errors[0][0].startswith("Unknown file"):
            pytest.fail(str(res.errors))  # pragma: no cover
        if manager.has_work:
            time.sleep(0.1)
        else:
            break
    else:  # pragma: no cover
        manager.kill_jobs()
        pytest.fail("jobs hanged")

    manager.writer.finish()
    if sys.platform == "darwin":
        time.sleep(2)  # pragma: no cover
    else:
        time.sleep(0.4)


# noinspection DuplicatedCode
class TestCalculationProcess:
    def test_mask_op(self, data_test_dir, tmpdir, mask_operation_plan):
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=os.path.dirname(file_path),
            result_prefix=tmpdir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=mask_operation_plan,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

    def test_one_file(self, data_test_dir, calculation_plan):
        process = CalculationProcess()
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component5.tif")
        calc = MocksCalculation(file_path)
        process.calculation = calc
        process.image = TiffImageReader.read_image(file_path)
        process.iterate_over(calculation_plan.execution_tree)
        assert len(process.measurement[0]) == 3

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_base(self, tmpdir, data_test_dir, monkeypatch, calculation_plan):
        monkeypatch.setattr(batch_backend, "CalculationProcess", MockCalculationProcess)
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        for file_path in file_paths:
            res = calc_process.do_calculation(FileCalculation(file_path, calc))
            assert isinstance(res, list)
            assert isinstance(res[0], ResponseData)

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline(self, tmpdir, data_test_dir, monkeypatch, calculation_plan):
        monkeypatch.setattr(batch_backend, "CalculationProcess", MockCalculationProcess)
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(3)
        manager.add_calculation(calc)
        wait_for_calculation(manager)
        assert os.path.exists(os.path.join(tmpdir, "test.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (8, 4)
        for i in range(8):
            assert os.path.basename(df.name.units[i]) == f"stack1_component{i+1}.tif"

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_long(self, tmpdir, data_test_dir, monkeypatch, calculation_plan_long):
        monkeypatch.setattr(batch_backend, "CalculationProcess", MockCalculationProcess)
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))[:4]
        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan_long,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(3)
        manager.add_calculation(calc)
        wait_for_calculation(manager)
        res_path = os.path.join(tmpdir, "test.xlsx")
        assert os.path.exists(res_path)
        df = pd.read_excel(res_path, index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (4, 20 * 3 + 1)
        data, err = LoadPlanExcel.load([res_path])
        assert not err
        assert str(data["test"]) == str(calculation_plan_long)

        df2 = pd.read_excel(res_path, header=[0, 1], engine=ENGINE, sheet_name="info test")
        assert df2.shape == (152, 3)

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_error(self, tmp_path, data_test_dir, monkeypatch, calculation_plan):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        file_pattern_copy = os.path.join(data_test_dir, "stack1_components", "stack1_component*.tif")
        file_paths = sorted(glob(file_pattern_copy))
        for el in file_paths:
            shutil.copy(el, data_dir)
        shutil.copy(data_dir / "stack1_component1.tif", data_dir / "stack1_component10.tif")
        file_pattern = os.path.join(data_dir, "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        result_dir = tmp_path / "result"
        result_dir.mkdir()

        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=str(data_dir),
            result_prefix=str(data_dir),
            measurement_file_path=os.path.join(result_dir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(3)
        manager.add_calculation(calc)
        wait_for_calculation(manager)

        assert os.path.exists(os.path.join(result_dir, "test.xlsx"))
        df = pd.read_excel(os.path.join(result_dir, "test.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (8, 4)
        for i in range(8):
            assert os.path.basename(df.name.units[i]) == f"stack1_component{i + 1}.tif"
        df2 = pd.read_excel(os.path.join(result_dir, "test.xlsx"), sheet_name="Errors", index_col=0, engine=ENGINE)
        assert df2.shape == (1, 2)
        str(df2.loc[0]["error description"]).startswith("[Errno 2]")

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_mask_project(self, tmpdir, data_test_dir, calculation_plan2):
        file_pattern = os.path.join(data_test_dir, "*nucleus.seg")
        file_paths = glob(file_pattern)
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test2.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan2,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(2)
        manager.add_calculation(calc)
        wait_for_calculation(manager)

        assert os.path.exists(os.path.join(tmpdir, "test2.xlsx"))
        df = pd.read_excel(os.path.join(tmpdir, "test2.xlsx"), index_col=0, header=[0, 1], engine=ENGINE)
        assert df.shape == (2, 4)

    def test_do_calculation(self, tmpdir, data_test_dir, calculation_plan3):
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan3,
            voxel_size=(1, 1, 1),
        )
        index, res = do_calculation((1, file_path), calc)
        assert index == 1
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

    @pytest.mark.parametrize(
        ("file_name", "root_type"),
        [
            (os.path.join("stack1_components", "stack1_component1.tif"), RootType.Image),
            ("stack1_component1_1.tgz", RootType.Image),
            ("stack1_component1_1.tgz", RootType.Project),
            ("test_nucleus_1_1.seg", RootType.Image),
            ("test_nucleus_1_1.seg", RootType.Mask_project),
        ],
    )
    @pytest.mark.parametrize("save_method", save_dict.values())
    def test_do_calculation_save(self, tmpdir, data_test_dir, file_name, root_type, save_method: SaveBase, simple_plan):
        save_desc = Save(
            suffix="_test",
            directory="",
            algorithm=save_method.get_name(),
            short_name=save_method.get_short_name(),
            values=save_method.get_default_values(),
        )
        plan = simple_plan(root_type, save_desc)
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

    def test_do_calculation_calculation_process(self, tmpdir, data_test_dir, calculation_plan3):
        file_path = os.path.join(data_test_dir, "stack1_components", "stack1_component1.tif")
        calc = Calculation(
            [file_path],
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan3,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert isinstance(res, list)
        assert isinstance(res[0], ResponseData)

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_component_split_no_process(self, tmpdir, data_test_dir, monkeypatch, calculation_plan3):
        monkeypatch.setattr(batch_backend, "CalculationProcess", MockCalculationProcess)
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = sorted(glob(file_pattern))
        assert os.path.basename(file_paths[0]) == "stack1_component1.tif"
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan3,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        for file_path in file_paths:
            res = calc_process.do_calculation(FileCalculation(file_path, calc))
            assert isinstance(res, list)
            assert isinstance(res[0], ResponseData)

    @pytest.mark.filterwarnings("ignore:This method will be removed")
    def test_full_pipeline_component_split(self, tmpdir, data_test_dir, calculation_plan3):
        file_pattern = os.path.join(data_test_dir, "stack1_components", "stack1_component*[0-9].tif")
        file_paths = glob(file_pattern)
        calc = Calculation(
            file_paths,
            base_prefix=data_test_dir,
            result_prefix=data_test_dir,
            measurement_file_path=os.path.join(tmpdir, "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan3,
            voxel_size=(1, 1, 1),
        )

        manager = CalculationManager()
        manager.set_number_of_workers(2)
        manager.add_calculation(calc)
        wait_for_calculation(manager)

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

    @pytest.mark.usefixtures("_prepare_mask_project_data")
    @pytest.mark.usefixtures("_register_dummy_extraction")
    def test_fail_single_mask_project(self, tmp_path, calculation_plan_dummy):
        file_path = str(tmp_path / "test.seg")
        calc = Calculation(
            [file_path],
            base_prefix=str(tmp_path),
            result_prefix=str(tmp_path),
            measurement_file_path=str(tmp_path / "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan_dummy,
            voxel_size=(1, 1, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path, calc))
        assert len(res) == 4
        assert sum(isinstance(x, ResponseData) for x in res) == 3
        assert isinstance(next(iter(dropwhile(lambda x: isinstance(x, ResponseData), res)))[0], ValueError)

    @pytest.mark.usefixtures("_prepare_spacing_data")
    @pytest.mark.usefixtures("_register_dummy_spacing")
    def test_spacing_overwrite(self, tmp_path, calculation_plan_dummy_spacing):
        file_path1 = str(tmp_path / "test1.tiff")
        file_path2 = str(tmp_path / "test2.tiff")
        calc = Calculation(
            [file_path1, file_path2],
            base_prefix=str(tmp_path),
            result_prefix=str(tmp_path),
            measurement_file_path=str(tmp_path / "test3.xlsx"),
            sheet_name="Sheet1",
            calculation_plan=calculation_plan_dummy_spacing,
            voxel_size=(3, 2, 1),
        )
        calc_process = CalculationProcess()
        res = calc_process.do_calculation(FileCalculation(file_path1, calc))
        assert len(res) == 1
        assert isinstance(res[0], ResponseData)
        calc.overwrite_voxel_size = True
        res = calc_process.do_calculation(FileCalculation(file_path2, calc))
        assert len(res) == 1
        assert isinstance(res[0], ResponseData)


class MockCalculationProcess(CalculationProcess):
    def do_calculation(self, calculation: FileCalculation):
        if os.path.basename(calculation.file_path) == "stack1_component1.tif":
            time.sleep(0.5)
        return super().do_calculation(calculation)


class TestSheetData:
    def test_create(self):
        cols = [("aa", "nm"), ("bb", "nm")]
        sheet_data = SheetData("test_name", cols)
        assert "test_name" in repr(sheet_data)
        assert str(cols) in repr(sheet_data)
        assert "wait_rows=0" in repr(sheet_data)

    def test_add_data(self):
        cols = [("aa", "nm"), ("bb", "nm")]
        sheet_data = SheetData("test_name", cols)
        with pytest.raises(ValueError, match="Wrong number of columns"):
            sheet_data.add_data(["aa", 1, 2, 3], None)

        with pytest.raises(ValueError, match="Wrong number of columns"):
            sheet_data.add_data(["aa", 1], None)

        sheet_data.add_data(["aa", 1, 2], None)
        assert "wait_rows=1" in repr(sheet_data)

        assert sheet_data.get_data_to_write()[0] == "test_name"
        assert "wait_rows=0" in repr(sheet_data)


def test_calculation_plan_serialize(calculation_plan_long):
    text = json.dumps(calculation_plan_long, cls=PartSegEncoder, indent=2)
    assert text.count("\n") == 7627
