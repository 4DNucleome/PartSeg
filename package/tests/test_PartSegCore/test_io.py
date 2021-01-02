import dataclasses
import json
import os.path
import re
import tarfile
from copy import deepcopy
from enum import Enum
from glob import glob

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
from PartSegCore.io_utils import SaveROIAsNumpy, UpdateLoadedMetadataBase
from PartSegCore.json_hooks import check_loaded_dict
from PartSegCore.mask.history_utils import create_history_element_from_segmentation_tuple
from PartSegCore.mask.io_functions import (
    LoadROIImage,
    LoadROIParameters,
    LoadSegmentation,
    LoadStackImage,
    LoadStackImageWithMask,
    MaskProjectTuple,
    SaveParametersJSON,
    SaveROI,
    save_components,
)
from PartSegCore.segmentation.algorithm_base import AdditionalLayerDescription
from PartSegCore.segmentation.noise_filtering import DimensionType
from PartSegCore.segmentation.segmentation_algorithm import ThresholdAlgorithm
from PartSegImage import Image
from PartSegImage.image import reduce_array


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
    segmentation = np.zeros(data.shape, dtype=np.uint8)
    segmentation[data == 70] = 1
    segmentation[data == 60] = 2
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
    segmentation = image.fit_array_to_image(segmentation.squeeze())
    return ProjectTuple(
        file_path="test_data.tiff",
        image=image,
        roi=segmentation,
        additional_layers={"denoised image": AdditionalLayerDescription(data=segmentation, layer_type="layer")},
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
    segmentation = np.zeros(data.shape, dtype=np.uint8)
    segmentation[data == 70] = 1
    segmentation[data == 60] = 2

    data = 100 - data
    image = Image(
        data, (10 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value], 5 / UNIT_SCALE[Units.nm.value]), ""
    )
    segmentation = image.fit_array_to_image(segmentation.squeeze())
    return ProjectTuple("test_data.tiff", image, segmentation, mask=mask)


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
        seg = LoadSegmentation.load([os.path.join(data_test_dir, "test_nucleus_1_1.seg")])
        assert isinstance(seg.image, str)
        assert seg.selected_components == [1, 3]
        assert os.path.exists(os.path.join(data_test_dir, seg.image))
        assert len(seg.roi_extraction_parameters) == 4
        assert os.path.basename(seg.image) == "test_nucleus.tif"

    def test_load_old_seg(self, data_test_dir):
        """
        For PartSeg 0.9.4 and older
        """
        seg = LoadSegmentation.load([os.path.join(data_test_dir, "test_nucleus.seg")])
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
        assert isinstance(seg.roi, np.ndarray)
        seg.image.fit_array_to_image(seg.roi)
        assert os.path.basename(seg.image.file_path) == "test_nucleus.tif"

    def test_load_seg_with_image(self, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        assert isinstance(seg.image, Image)
        assert seg.selected_components == [1, 3]
        assert isinstance(seg.roi, np.ndarray)
        seg.image.fit_array_to_image(seg.roi)
        assert os.path.basename(seg.image.file_path) == "test_nucleus.tif"

    def test_save_segmentation(self, tmpdir, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        SaveROI.save(os.path.join(tmpdir, "segmentation.seg"), seg, {"relative_path": False})
        assert os.path.exists(os.path.join(tmpdir, "segmentation.seg"))
        os.makedirs(os.path.join(tmpdir, "seg_save"))
        save_components(seg.image, seg.selected_components, seg.roi, os.path.join(tmpdir, "seg_save"))
        assert os.path.isdir(os.path.join(tmpdir, "seg_save"))
        assert len(glob(os.path.join(tmpdir, "seg_save", "*"))) == 4
        seg2 = LoadSegmentation.load([os.path.join(tmpdir, "segmentation.seg")])
        assert seg2 is not None

    def test_save_segmentation_without_image(self, tmpdir, data_test_dir):
        seg = LoadROIImage.load(
            [os.path.join(data_test_dir, "test_nucleus_1_1.seg")], metadata={"default_spacing": (1, 1, 1)}
        )
        seg_clean = dataclasses.replace(seg, image=None, roi=reduce_array(seg.roi))
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
            image_data.image.file_path, image_data.image, None, res.roi, list(range(1, num)), data_dict
        )

        SaveROI.save(os.path.join(tmpdir, "segmentation2.seg"), to_save, {"relative_path": False})
        seg2 = LoadSegmentation.load([os.path.join(tmpdir, "segmentation2.seg")])
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
            mask=stack_segmentation1.roi,
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
            mask=stack_segmentation1.roi,
            image=stack_segmentation1.image.substitute(file_path=image_location),
            file_path=image_location,
        )
        SaveROI.save(tmp_path / "test1.seg", seg2, {"relative_path": False})
        res = LoadSegmentation.load([tmp_path / "test1.seg"])
        assert res.image == str(image_location)
        assert res.mask is not None
        assert len(res.history) == 1
        assert res.history[0].mask_property == mask_property
        cmp_dict = {str(k): v for k, v in stack_segmentation1.roi_extraction_parameters.items()}
        assert str(res.history[0].segmentation_parameters["parameters"]) == str(cmp_dict)


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
        assert (1,) + arr.shape == analysis_project.roi.shape
        assert steps == (5, 5, 10)

        parameters = {"channel": 0, "separated_objects": True, "clip": False, "units": Units.nm, "reverse": False}
        SaveCmap.save(os.path.join(tmpdir, "test2.cmap"), analysis_project, parameters)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp1.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 15, 15)
        assert tuple(np.max(points, axis=1)) == (34, 34, 84)
        assert (1,) + arr.shape == analysis_project.roi.shape
        assert steps == (5, 5, 10)
        arr, steps = self.read_cmap(os.path.join(tmpdir, "test2_comp2.cmap"))
        points = np.nonzero(arr)
        assert tuple(np.min(points, axis=1)) == (15, 55, 15)
        assert tuple(np.max(points, axis=1)) == (34, 84, 84)
        assert (1,) + arr.shape == analysis_project.roi.shape
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
        assert (1,) + arr.shape == analysis_project_reversed.roi.shape
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
        assert np.max(load_data.roi) == 2
        # TODO add more checks

    def test_load_project(self, data_test_dir):
        load_data = LoadProject.load([os.path.join(data_test_dir, "stack1_component1_1.tgz")])
        assert np.max(load_data.roi) == 2
        # TODO add more checks

    def test_save_project(self, tmpdir, analysis_project):
        SaveProject.save(os.path.join(tmpdir, "test1.tgz"), analysis_project)
        assert os.path.exists(os.path.join(tmpdir, "test1.tgz"))
        LoadProject.load([os.path.join(tmpdir, "test1.tgz")])
        # TODO add more

    def test_save_tiff(self, tmpdir, analysis_project):
        SaveAsTiff.save(os.path.join(tmpdir, "test1.tiff"), analysis_project)
        array = tifffile.imread(os.path.join(tmpdir, "test1.tiff"))
        assert analysis_project.roi.shape == (1,) + array.shape

    def test_save_numpy(self, tmpdir, analysis_project):
        parameters = {"squeeze": False}
        SaveAsNumpy.save(os.path.join(tmpdir, "test1.npy"), analysis_project, parameters)
        array = np.load(os.path.join(tmpdir, "test1.npy"))
        assert array.shape == analysis_project.image.shape
        assert np.all(array == analysis_project.image.get_data())
        parameters = {"squeeze": True}
        SaveAsNumpy.save(os.path.join(tmpdir, "test2.npy"), analysis_project, parameters)
        array = np.load(os.path.join(tmpdir, "test2.npy"))
        assert (1,) + array.shape == analysis_project.roi.shape
        assert np.all(array == analysis_project.image.get_data().squeeze())

    def test_save_segmentation_numpy(self, tmpdir, analysis_project):
        SaveROIAsNumpy.save(os.path.join(tmpdir, "test1.npy"), analysis_project)
        array = np.load(os.path.join(tmpdir, "test1.npy"))
        assert np.all(array == analysis_project.roi)


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
