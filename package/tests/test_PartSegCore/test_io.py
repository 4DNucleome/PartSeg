import dataclasses
import json
import os.path
import re
import tarfile
from copy import deepcopy
from enum import Enum
from glob import glob
from typing import Type

import h5py
import numpy as np
import pandas as pd
import pytest
import tifffile

from PartSegCore import UNIT_SCALE, Units
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import ProjectTuple
from PartSegCore.analysis.calculation_plan import CalculationPlan, MaskSuffix, MeasurementCalculate
from PartSegCore.analysis.load_functions import LoadProject, UpdateLoadedMetadataAnalysis
from PartSegCore.analysis.measurement_base import Leaf, MeasurementEntry
from PartSegCore.analysis.measurement_calculation import MEASUREMENT_DICT, MeasurementProfile
from PartSegCore.analysis.save_functions import SaveAsNumpy, SaveAsTiff, SaveCmap, SaveProject, SaveXYZ
from PartSegCore.analysis.save_hooks import PartEncoder, part_hook
from PartSegCore.class_generator import enum_register
from PartSegCore.image_operations import RadiusType
from PartSegCore.io_utils import LoadBase, SaveBase, SaveROIAsNumpy, UpdateLoadedMetadataBase
from PartSegCore.json_hooks import check_loaded_dict
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import (
    LoadROI,
    LoadROIImage,
    LoadROIParameters,
    LoadStackImage,
    LoadStackImageWithMask,
    MaskProjectTuple,
    SaveParametersJSON,
    SaveROI,
    save_components,
)
from PartSegCore.mask_create import MaskProperty
from PartSegCore.project_info import HistoryElement, ProjectInfoBase
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.segmentation.segmentation_algorithm import ThresholdAlgorithm
from PartSegImage import Image


@pytest.fixture(scope="module")
def analysis_project() -> ProjectTuple:
    data = np.zeros((1, 50, 100, 100, 1), dtype=np.uint16)
    data[0, 10:40, 10:40, 10:90] = 50
    data[0, 10:40, 50:90, 10:90] = 50
    data[0, 15:35, 15:35, 15:85] = 70
    data[0, 15:35, 55:85, 15:85] = 60
    data[0, 10:40, 40:50, 10:90] = 40
    image = Image(
        data, (10 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value]), ""
    )
    mask = data[0, ..., 0] > 0
    roi = np.zeros(data.shape, dtype=np.uint8)
    roi[data == 70] = 1
    roi[data == 60] = 2
    algorithm_parameters = {
        "algorithm_name": "Lower Threshold",
        "values": {
            "threshold": 60,
            "channel": 0,
            "noise_filtering": {"name": "None", "values": {}},
            "minimum_size": 10,
            "side_connection": False,
        },
    }
    roi_info = ROIInfo(roi.squeeze()).fit_to_image(image)
    return ProjectTuple(
        file_path="test_data.tiff",
        image=image,
        roi_info=roi_info,
        additional_layers={"denoised image": AdditionalLayerDescription(data=roi, layer_type="layer")},
        mask=mask,
        algorithm_parameters=algorithm_parameters,
    )


@pytest.fixture(scope="module")
def analysis_project_reversed() -> ProjectTuple:
    data = np.zeros((1, 50, 100, 100, 1), dtype=np.uint16)
    data[0, 10:40, 10:40, 10:90] = 50
    data[0, 10:40, 50:90, 10:90] = 50
    data[0, 15:35, 15:35, 15:85] = 70
    data[0, 15:35, 55:85, 15:85] = 60
    data[0, 10:40, 40:50, 10:90] = 40
    mask = data[0] > 0
    roi = np.zeros(data.shape, dtype=np.uint8)
    roi[data == 70] = 1
    roi[data == 60] = 2

    data = 100 - data
    image = Image(
        data, (10 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value]), ""
    )
    roi_info = ROIInfo(roi.squeeze()).fit_to_image(image)
    return ProjectTuple("test_data.tiff", image, roi_info=roi_info, mask=mask)


@pytest.fixture
def mask_prop():
    return MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, False, False)


class TestHistoryElement:
    def test_create(self, mask_prop):
        roi_info = ROIInfo(np.zeros((10, 10), dtype=np.uint8))
        elem = HistoryElement.create(roi_info, None, {}, mask_prop)
        assert elem.mask_property == mask_prop
        assert elem.roi_extraction_parameters == {}
        param = {"a": 1, "b": 2}
        elem2 = HistoryElement.create(roi_info, None, param, mask_prop)
        assert elem2.roi_extraction_parameters == param
        roi_info2, mask = elem2.get_roi_info_and_mask()
        assert np.all(roi_info2.roi == 0)
        assert mask is None

    def test_mask(self, mask_prop):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[1:-1, 1:-1] = 2
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2] = 4
        roi_info = ROIInfo(mask2)
        elem = HistoryElement.create(roi_info, mask, {}, mask_prop)
        roi_info2, mask2 = elem.get_roi_info_and_mask()
        assert np.all(mask == mask2)
        assert np.all(roi_info.roi == roi_info2.roi)

    def test_additional(self, mask_prop):
        data = np.zeros((10, 10), dtype=np.uint8)
        data[1:-1, 1:-1] = 2
        alternative = {}
        for i in range(4):
            alternative_array = np.copy(data)
            alternative_array[alternative_array > 0] = i + 2
            alternative[f"add{i}"] = alternative_array
        roi_info = ROIInfo(data, alternative=alternative)
        elem = HistoryElement.create(roi_info, None, {}, mask_prop)
        roi_info2, mask2 = elem.get_roi_info_and_mask()
        assert np.all(roi_info2.roi == roi_info.roi)
        assert mask2 is None
        assert len(roi_info.alternative) == 4
        assert set(roi_info.alternative) == {f"add{i}" for i in range(4)}
        for i in range(4):
            arr = roi_info.alternative[f"add{i}"]
            assert np.all(arr[arr > 0] == i + 2)

    def test_annotations(self, mask_prop):
        data = np.zeros((10, 10), dtype=np.uint8)
        data[1:5, 1:5] = 1
        data[5:-1, 1:5] = 2
        data[1:5, 5:-1] = 3
        data[5:-1, 5:-1] = 4
        annotations = {1: "a", 2: "b", 3: "c", 4: "d"}
        roi_info = ROIInfo(data, annotations=annotations)
        elem = HistoryElement.create(roi_info, None, {}, mask_prop)
        roi_info2, mask2 = elem.get_roi_info_and_mask()
        assert mask2 is None
        assert np.all(roi_info2.roi == roi_info.roi)


class TestSaveHistory:
    def test_save_roi_info_project_tuple(self, analysis_segmentation2, tmp_path):
        self.perform_roi_info_test(analysis_segmentation2, tmp_path, SaveProject, LoadProject)

    def test_save_roi_info_mask_project(self, stack_segmentation2, tmp_path):
        self.perform_roi_info_test(stack_segmentation2, tmp_path, SaveROI, LoadROI)

    def perform_roi_info_test(self, project, save_path, save_method: Type[SaveBase], load_method: Type[LoadBase]):
        alt1 = np.copy(project.roi_info.roi)
        alt1[alt1 > 0] += 3
        roi_info = ROIInfo(
            roi=project.roi_info.roi, annotations={i: f"a{i}" for i in range(1, 5)}, alternative={"test": alt1}
        )
        proj = dataclasses.replace(project, roi_info=roi_info)
        save_method.save(save_path / "data.tgz", proj, SaveROI.get_default_values())
        proj2 = load_method.load([save_path / "data.tgz"])
        assert np.all(proj2.roi_info.roi == project.roi_info.roi)
        assert set(proj2.roi_info.annotations) == {1, 2, 3, 4}
        assert proj2.roi_info.annotations == {i: f"a{i}" for i in range(1, 5)}
        assert "test" in proj2.roi_info.alternative
        assert np.all(proj2.roi_info.alternative["test"] == alt1)

    def test_save_roi_info_history_project_tuple(self, analysis_segmentation2, mask_property, tmp_path):
        self.perform_roi_info_history_test(analysis_segmentation2, tmp_path, mask_property, SaveProject, LoadProject)

    def test_save_roi_info_history_mask_project(self, stack_segmentation2, mask_property, tmp_path):
        self.perform_roi_info_history_test(stack_segmentation2, tmp_path, mask_property, SaveROI, LoadROI)

    def perform_roi_info_history_test(
        self, project, save_path, mask_property, save_method: Type[SaveBase], load_method: Type[LoadBase]
    ):
        alt1 = np.copy(project.roi_info.roi)
        alt1[alt1 > 0] += 3
        roi_info = ROIInfo(
            roi=project.roi_info.roi, annotations={i: f"a{i}" for i in range(1, 5)}, alternative={"test": alt1}
        )
        history = []
        for i in range(3):
            alt2 = np.copy(alt1)
            alt2[alt2 > 0] = i + 5
            roi_info2 = ROIInfo(
                roi=project.roi_info.roi,
                annotations={i: f"a{i}_{j}" for j in range(1, 5)},
                alternative={f"test{i}": alt2},
            )
            history.append(
                HistoryElement.create(
                    roi_info2, alt1, {"algorithm_name": f"task_{i}", "values": {"a": 1}}, mask_property
                )
            )
        proj = dataclasses.replace(project, roi_info=roi_info, history=history)
        save_method.save(save_path / "data.tgz", proj, SaveROI.get_default_values())
        proj2: ProjectInfoBase = load_method.load([save_path / "data.tgz"])
        assert np.all(proj2.roi_info.roi == project.roi_info.roi)
        assert set(proj2.roi_info.annotations) == {1, 2, 3, 4}
        assert proj2.roi_info.annotations == {i: f"a{i}" for i in range(1, 5)}
        assert "test" in proj2.roi_info.alternative
        assert np.all(proj2.roi_info.alternative["test"] == alt1)
        assert len(proj2.history) == 3
        for i in range(3):
            roi_info3, mask2 = proj2.history[i].get_roi_info_and_mask()
            assert np.all(mask2 == alt1)
            assert set(roi_info3.alternative) == {f"test{i}"}
            assert np.all(roi_info3.alternative[f"test{i}"][alt1 > 0] == i + 5)
            assert np.all(roi_info3.alternative[f"test{i}"][alt1 == 0] == 0)
            assert roi_info3.annotations == {i: f"a{i}_{j}" for j in range(1, 5)}
            assert proj2.history[i].roi_extraction_parameters == {"algorithm_name": f"task_{i}", "values": {"a": 1}}


class TestJsonLoad:
    def test_profile_load(self, data_test_dir):
        profile_path = os.path.join(data_test_dir, "segment_profile_test.json")
        # noinspection PyBroadException
        try:
            with open(profile_path) as ff:
                data = json.load(ff, object_hook=part_hook)
            assert check_loaded_dict(data)
        except Exception:  # pylint: disable=W0703
            pytest.fail("Fail in loading profile")

    def test_measure_load(self, data_test_dir):
        profile_path = os.path.join(data_test_dir, "measurements_profile_test.json")
        # noinspection PyBroadException
        try:
            with open(profile_path) as ff:
                data = json.load(ff, object_hook=part_hook)
            assert check_loaded_dict(data)
        except Exception:  # pylint: disable=W0703
            pytest.fail("Fail in loading profile")

    def test_json_dump(self):
        with pytest.raises(TypeError):
            json.dumps(DimensionType.Layer)
        data_string = json.dumps(DimensionType.Layer, cls=PartEncoder)
        assert re.search('"__Enum__":[^,}]+[,}]', data_string) is not None
        assert re.search('"__subtype__":[^,}]+[,}]', data_string) is not None
        assert re.search('"value":[^,}]+[,}]', data_string) is not None

    def test_json_load(self):
        class Test(Enum):
            test0 = 0
            test1 = 1
            test2 = 2

        test_json = json.dumps(Test.test0, cls=PartEncoder)

        assert not check_loaded_dict(json.loads(test_json, object_hook=part_hook))

        enum_register.register_class(Test)
        assert isinstance(json.loads(test_json, object_hook=part_hook), Test)

    def test_modernize_0_9_2_3(self, bundle_test_dir):
        file_path = os.path.join(bundle_test_dir, "segment_profile_0.9.2.3.json")
        assert os.path.exists(file_path)
        data = UpdateLoadedMetadataBase.load_json_data(file_path)
        assert "noise_filtering" in data["test_0.9.2.3"].values
        assert "dimension_type" in data["test_0.9.2.3"].values["noise_filtering"]["values"]
        file_path = os.path.join(bundle_test_dir, "calculation_plan_0.9.2.3.json")
        data = UpdateLoadedMetadataAnalysis.load_json_data(file_path)

    def test_update_name(self):
        data = UpdateLoadedMetadataAnalysis.load_json_data(update_name_json)
        print(data)
        mp = data["problematic set"]
        assert isinstance(mp, MeasurementProfile)
        assert isinstance(mp.chosen_fields[0], MeasurementEntry)
        assert isinstance(mp.chosen_fields[0].calculation_tree, Leaf)
        assert mp.chosen_fields[0].calculation_tree.name == "Pixel brightness sum"
        assert mp.chosen_fields[1].calculation_tree.name == "Components number"

    def test_load_workflow(self, bundle_test_dir):
        data = UpdateLoadedMetadataAnalysis.load_json_data(os.path.join(bundle_test_dir, "workflow.json"))
        plan = data["workflow"]
        assert isinstance(plan, CalculationPlan)
        mask_step = plan.execution_tree.children[0]
        assert isinstance(mask_step.operation, MaskSuffix)
        segmentation_step = mask_step.children[0]
        assert isinstance(segmentation_step.operation, ROIExtractionProfile)
        measurement_step = segmentation_step.children[0]
        assert isinstance(measurement_step.operation, MeasurementCalculate)
        assert measurement_step.children == []
        measurement_profile = measurement_step.operation.measurement_profile
        assert isinstance(measurement_profile, MeasurementProfile)
        for entry in measurement_profile.chosen_fields:
            assert entry.calculation_tree.name in MEASUREMENT_DICT


class TestSegmentationMask:
    def test_load_seg(self, data_test_dir):
        seg = LoadROI.load([os.path.join(data_test_dir, "test_nucleus_1_1.seg")])
        assert isinstance(seg.image, str)
        assert seg.selected_components == [1, 3]
        assert os.path.exists(os.path.join(data_test_dir, seg.image))
        assert len(seg.roi_extraction_parameters) == 4
        assert os.path.basename(seg.image) == "test_nucleus.tif"

    def test_load_old_seg(self, data_test_dir):
        """
        For PartSeg 0.9.4 and older
        """
        seg = LoadROI.load([os.path.join(data_test_dir, "test_nucleus.seg")])
        assert isinstance(seg.image, str)
        assert seg.selected_components == [1, 3]
        assert os.path.exists(os.path.join(data_test_dir, seg.image))
        assert os.path.basename(seg.image) == "test_nucleus.tif"

    def test_load_old_seg_with_image(self, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        assert isinstance(seg.image, Image)
        assert seg.selected_components == [1, 3]
        assert isinstance(seg.roi_info.roi, np.ndarray)
        seg.image.fit_array_to_image(seg.roi_info.roi)
        assert os.path.basename(seg.image.file_path) == "test_nucleus.tif"

    def test_load_seg_with_image(self, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        assert isinstance(seg.image, Image)
        assert seg.selected_components == [1, 3]
        assert isinstance(seg.roi_info.roi, np.ndarray)
        seg.image.fit_array_to_image(seg.roi_info.roi)
        assert os.path.basename(seg.image.file_path) == "test_nucleus.tif"

    def test_save_segmentation(self, tmpdir, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        SaveROI.save(os.path.join(tmpdir, "segmentation.seg"), seg, {"relative_path": False})
        assert os.path.exists(os.path.join(tmpdir, "segmentation.seg"))
        os.makedirs(os.path.join(tmpdir, "seg_save"))
        save_components(seg.image, seg.selected_components, os.path.join(tmpdir, "seg_save"), seg.roi_info)
        assert os.path.isdir(os.path.join(tmpdir, "seg_save"))
        assert len(glob(os.path.join(tmpdir, "seg_save", "*"))) == 4
        seg2 = LoadROI.load([os.path.join(tmpdir, "segmentation.seg")])
        assert seg2 is not None

    def test_save_segmentation_without_image(self, tmpdir, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        seg_clean = dataclasses.replace(seg, image=None, roi_info=seg.roi_info)
        SaveROI.save(os.path.join(tmpdir, "segmentation.seg"), seg_clean, {"relative_path": False})
        SaveROI.save(
            os.path.join(tmpdir, "segmentation1.seg"),
            seg_clean,
            {"relative_path": False, "spacing": (210 * 10 ** -6, 70 * 10 ** -6, 70 * 10 ** -6)},
        )

    def test_loading_new_segmentation(self, tmpdir, data_test_dir):
        image_data = LoadStackImage.load([os.path.join(data_test_dir, "test_nucleus.tif")])
        algorithm = ThresholdAlgorithm()
        algorithm.set_image(image_data.image)
        param = algorithm.get_default_values()
        param["channel"] = 0
        algorithm.set_parameters(**param)
        res = algorithm.calculation_run(lambda x, y: None)
        num = np.max(res.roi) + 1
        data_dict = {str(i): deepcopy(res.parameters) for i in range(1, num)}

        to_save = MaskProjectTuple(
            image_data.image.file_path, image_data.image, None, res.roi_info, list(range(1, num)), data_dict
        )

        SaveROI.save(os.path.join(tmpdir, "segmentation2.seg"), to_save, {"relative_path": False})
        seg2 = LoadROI.load([os.path.join(tmpdir, "segmentation2.seg")])
        assert seg2 is not None

    def test_load_mask(self, data_test_dir):
        res = LoadStackImage.load([os.path.join(data_test_dir, "test_nucleus.tif")])
        assert res.mask is None
        res = LoadStackImageWithMask.load(
            [os.path.join(data_test_dir, "test_nucleus.tif"), os.path.join(data_test_dir, "test_nucleus_mask.tif")]
        )
        assert res.image.mask is not None
        assert res.mask is not None

    def test_save_project_with_history(self, tmp_path, stack_segmentation1, mask_property):
        SaveROI.save(tmp_path / "test1.seg", stack_segmentation1, {"relative_path": False})
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
            mask=stack_segmentation1.roi_info.roi,
        )
        SaveROI.save(tmp_path / "test1.seg", seg2, {"relative_path": False})
        with tarfile.open(tmp_path / "test1.seg", "r") as tf:
            tf.getmember("mask.tif")
            tf.getmember("segmentation.tif")
            tf.getmember("history/history.json")
            tf.getmember("history/arrays_0.npz")

    def test_load_project_with_history(self, tmp_path, stack_segmentation1, mask_property):
        image_location = tmp_path / "test1.tif"
        SaveAsTiff.save(image_location, stack_segmentation1)
        seg2 = dataclasses.replace(
            stack_segmentation1,
            history=[create_history_element_from_segmentation_tuple(stack_segmentation1, mask_property)],
            selected_components=[1],
            mask=stack_segmentation1.roi_info.roi,
            image=stack_segmentation1.image.substitute(file_path=image_location),
            file_path=image_location,
        )
        SaveROI.save(tmp_path / "test1.seg", seg2, {"relative_path": False})
        res = LoadROI.load([tmp_path / "test1.seg"])
        assert res.image == str(image_location)
        assert res.mask is not None
        assert len(res.history) == 1
        assert res.history[0].mask_property == mask_property
        cmp_dict = {str(k): v for k, v in stack_segmentation1.roi_extraction_parameters.items()}
        assert str(res.history[0].roi_extraction_parameters["parameters"]) == str(cmp_dict)


class TestSaveFunctions:
    @staticmethod
    def read_cmap(file_path):
        with h5py.File(file_path, "r") as fp:
            arr = np.array(fp.get("Chimera/image1/data_zyx"))
            steps = tuple(map(lambda x: int(x + 0.5), fp.get("Chimera/image1").attrs["step"]))
            return arr, steps

    def test_save_cmap(self, tmpdir, analysis_project):
        parameters = {"channel": 0, "separated_objects": False, "clip": False, "units": Units.nm, "reverse": False}
        SaveCmap.save(os.path.join(tmpdir, "test1.cmap"), analysis_project, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 15, 15)
        assert tuple(np.max(points, axis=1)) == (34, 84, 84)
        assert (1,) + arr.shape == analysis_project.roi_info.roi.shape
        assert steps == (5, 5, 10)

        parameters = {"channel": 0, "separated_objects": True, "clip": False, "units": Units.nm, "reverse": False}
        SaveCmap.save(os.path.join(tmpdir, "test2.cmap"), analysis_project, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 15, 15)
        assert tuple(np.max(points, axis=1)) == (34, 34, 84)
        assert (1,) + arr.shape == analysis_project.roi_info.roi.shape
        assert steps == (5, 5, 10)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp2.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 55, 15)
        assert tuple(np.max(points, axis=1)) == (34, 84, 84)
        assert (1,) + arr.shape == analysis_project.roi_info.roi.shape
        assert steps == (5, 5, 10)

        parameters = {"channel": 0, "separated_objects": False, "clip": True, "units": Units.nm, "reverse": False}
        SaveCmap.save(os.path.join(tmpdir, "test3.cmap"), analysis_project, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test3.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (0, 0, 0)
        assert tuple(np.max(points, axis=1)) == (19, 69, 69)
        assert arr.shape == (20, 70, 70)
        assert steps == (5, 5, 10)

        parameters = {"channel": 0, "separated_objects": True, "clip": True, "units": Units.nm, "reverse": False}
        SaveCmap.save(os.path.join(tmpdir, "test4.cmap"), analysis_project, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test4_comp1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (0, 0, 0)
        assert tuple(np.max(points, axis=1)) == (19, 19, 69)
        assert arr.shape == (20, 70, 70)
        assert steps == (5, 5, 10)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test4_comp2.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (0, 40, 0)
        assert tuple(np.max(points, axis=1)) == (19, 69, 69)
        assert arr.shape == (20, 70, 70)
        assert steps == (5, 5, 10)

    def test_save_cmap_reversed(self, tmpdir, analysis_project_reversed):
        parameters = {"channel": 0, "separated_objects": False, "clip": False, "units": Units.nm, "reverse": True}
        SaveCmap.save(os.path.join(tmpdir, "test1.cmap"), analysis_project_reversed, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 15, 15)
        assert tuple(np.max(points, axis=1)) == (34, 84, 84)
        assert (1,) + arr.shape == analysis_project_reversed.roi_info.roi.shape
        assert steps == (5, 5, 10)

        parameters = {"channel": 0, "separated_objects": True, "clip": True, "units": Units.nm, "reverse": True}
        SaveCmap.save(os.path.join(tmpdir, "test2.cmap"), analysis_project_reversed, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (0, 0, 0)
        assert tuple(np.max(points, axis=1)) == (19, 19, 69)
        assert arr.shape == (20, 70, 70)
        assert steps == (5, 5, 10)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp2.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (0, 40, 0)
        assert tuple(np.max(points, axis=1)) == (19, 69, 69)
        assert arr.shape == (20, 70, 70)
        assert steps == (5, 5, 10)

    @pytest.mark.parametrize("separated_objects", [True, False])
    @pytest.mark.parametrize("clip", [True, False])
    def test_save_xyz(self, tmpdir, analysis_project, separated_objects, clip):
        parameters = {"channel": 0, "separated_objects": separated_objects, "clip": clip}
        SaveXYZ.save(os.path.join(tmpdir, "test1.xyz"), analysis_project, parameters)
        array = pd.read_csv(os.path.join(tmpdir, "test1.xyz"), dtype=np.uint16, sep=" ")
        shift = (15, 15, 15, 0) if clip else (0, 0, 0, 0)

        assert np.all(np.min(array, axis=0) == np.subtract((15, 15, 15, 60), shift))
        assert np.all(np.max(array, axis=0) == np.subtract((84, 84, 34, 70), shift))
        if separated_objects:
            array = pd.read_csv(os.path.join(tmpdir, "test1_part1.xyz"), dtype=np.uint16, sep=" ")
            assert np.all(np.min(array, axis=0) == np.subtract((15, 15, 15, 70), shift))
            assert np.all(np.max(array, axis=0) == np.subtract((84, 34, 34, 70), shift))
            array = pd.read_csv(os.path.join(tmpdir, "test1_part2.xyz"), dtype=np.uint16, sep=" ")
            assert np.all(np.min(array, axis=0) == np.subtract((15, 55, 15, 60), shift))
            assert np.all(np.max(array, axis=0) == np.subtract((84, 84, 34, 60), shift))

    def test_load_old_project(self, data_test_dir):
        load_data = LoadProject.load([os.path.join(data_test_dir, "stack1_component1.tgz")])
        assert np.max(load_data.roi_info.roi) == 2
        # TODO add more checks

    def test_load_project(self, data_test_dir):
        load_data = LoadProject.load([os.path.join(data_test_dir, "stack1_component1_1.tgz")])
        assert np.max(load_data.roi_info.roi) == 2
        # TODO add more checks

    def test_save_project(self, tmpdir, analysis_project):
        SaveProject.save(os.path.join(tmpdir, "test1.tgz"), analysis_project)
        assert os.path.exists(os.path.join(tmpdir, "test1.tgz"))
        LoadProject.load([os.path.join(tmpdir, "test1.tgz")])
        # TODO add more

    def test_save_tiff(self, tmpdir, analysis_project):
        SaveAsTiff.save(os.path.join(tmpdir, "test1.tiff"), analysis_project)
        array = tifffile.imread(os.path.join(tmpdir, "test1.tiff"))
        assert analysis_project.roi_info.roi.shape == (1,) + array.shape

    def test_save_numpy(self, tmpdir, analysis_project):
        parameters = {"squeeze": False}
        SaveAsNumpy.save(os.path.join(tmpdir, "test1.npy"), analysis_project, parameters)
        array = np.load(os.path.join(tmpdir, "test1.npy"))
        assert array.shape == analysis_project.image.shape
        assert np.all(array == analysis_project.image.get_data())
        parameters = {"squeeze": True}
        SaveAsNumpy.save(os.path.join(tmpdir, "test2.npy"), analysis_project, parameters)
        array = np.load(os.path.join(tmpdir, "test2.npy"))
        assert (1,) + array.shape == analysis_project.roi_info.roi.shape
        assert np.all(array == analysis_project.image.get_data().squeeze())

    def test_save_segmentation_numpy(self, tmpdir, analysis_project):
        SaveROIAsNumpy.save(os.path.join(tmpdir, "test1.npy"), analysis_project)
        array = np.load(os.path.join(tmpdir, "test1.npy"))
        assert np.all(array == analysis_project.roi_info.roi)


def test_json_parameters_mask(stack_segmentation1, tmp_path):
    SaveParametersJSON.save(tmp_path / "test.json", stack_segmentation1)
    load_param = LoadROIParameters.load([tmp_path / "test.json"])
    assert len(load_param.roi_extraction_parameters) == 4


update_name_json = """
{"problematic set": {
      "__MeasurementProfile__": true,
      "name": "problematic set",
      "chosen_fields": [
        {
          "__Serializable__": true,
          "__subtype__": "PartSegCore.analysis.measurement_base.MeasurementEntry",
          "name": "Segmentation Pixel Brightness Sum",
          "calculation_tree": {
            "__Serializable__": true,
            "__subtype__": "PartSegCore.analysis.measurement_base.Leaf",
            "name": "Pixel Brightness Sum",
            "dict": {},
            "power": 1.0,
            "area": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.AreaType",
              "value": 1
            },
            "per_component": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.PerComponent",
              "value": 1
            },
            "channel": null
          }
        },
        {
          "__Serializable__": true,
          "__subtype__": "PartSegCore.analysis.measurement_base.MeasurementEntry",
          "name": "Segmentation Components Number",
          "calculation_tree": {
            "__Serializable__": true,
            "__subtype__": "PartSegCore.analysis.measurement_base.Leaf",
            "name": "Components Number",
            "dict": {},
            "power": 1.0,
            "area": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.AreaType",
              "value": 1
            },
            "per_component": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.PerComponent",
              "value": 1
            },
            "channel": null
          }
        },
        {
          "__Serializable__": true,
          "__subtype__": "PartSegCore.analysis.measurement_base.MeasurementEntry",
          "name": "Segmentation Diameter",
          "calculation_tree": {
            "__Serializable__": true,
            "__subtype__": "PartSegCore.analysis.measurement_base.Leaf",
            "name": "Diameter",
            "dict": {},
            "power": 1.0,
            "area": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.AreaType",
              "value": 1
            },
            "per_component": {
              "__Enum__": true,
              "__subtype__": "PartSegCore.analysis.measurement_base.PerComponent",
              "value": 1
            },
            "channel": null
          }
        }
      ],
      "name_prefix": ""
    }
  }
"""
