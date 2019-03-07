import shutil
import tempfile
from enum import Enum
import os.path
import numpy as np
import pytest
import json
import re
from glob import glob

from PartSeg.tiff_image import ImageReader, Image
from PartSeg.utils.global_settings import static_file_folder
from PartSeg.utils.analysis.save_hooks import PartEncoder, part_hook
from PartSeg.utils.json_hooks import check_loaded_dict
from PartSeg.utils.segmentation.noise_filtering import GaussType
from PartSeg.utils.class_generator import enum_register
from PartSeg.utils.mask.io_functions import LoadSegmentation, SaveSegmentation, LoadSegmentationImage, save_components

from help_fun import get_test_dir

tmp_dir = ""


def setup_module():
    global tmp_dir
    tmp_dir = tempfile.mkdtemp()


def teardown_module():
    global tmp_dir
    shutil.rmtree(tmp_dir)


class TestImageClass:
    def test_image_read(self):
        image = ImageReader.read_image(os.path.join(static_file_folder, "initial_images", "stack.tif"))
        assert isinstance(image, Image)

    def test_image_mask(self):
        Image(np.zeros((1, 10, 50, 50, 4)), [5, 5, 5], mask=np.zeros((10, 50, 50)))
        Image(np.zeros((1, 10, 50, 50, 4)), [5, 5, 5], mask=np.zeros((1, 10, 50, 50)))
        with pytest.raises(ValueError):
            Image(np.zeros((1, 10, 50, 50, 4)), [5, 5, 5], mask=np.zeros((1, 10, 50, 40)))
        with pytest.raises(ValueError):
            Image(np.zeros((1, 10, 50, 50, 4)), [5, 5, 5], mask=np.zeros((1, 10, 50, 50, 4)))

    def test_read_with_mask(self):
        test_dir = get_test_dir()
        image = ImageReader.read_image(os.path.join(test_dir, "stack1_components", "stack1_component1.tif"),
                                       os.path.join(test_dir, "stack1_components", "stack1_component1_mask.tif"))
        assert isinstance(image, Image)
        with pytest.raises(ValueError):
            ImageReader.read_image(os.path.join(test_dir, "stack1_components", "stack1_component1.tif"),
                                   os.path.join(test_dir, "stack1_components", "stack1_component2_mask.tif"))

    def test_lsm_read(self):
        test_dir = get_test_dir()
        image1 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.lsm"))
        image2 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.tif"))
        data = np.load(os.path.join(test_dir, "test_lsm.npy"))
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())

    def test_ome_read(self):  # error in tifffile
        test_dir = get_test_dir()
        image1 = ImageReader.read_image(os.path.join(test_dir, "test_lsm2.tif"))
        image2 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.tif"))
        data = np.load(os.path.join(test_dir, "test_lsm.npy"))
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())


class TestJsonLoad:
    def test_profile_load(self):
        profile_path = os.path.join(get_test_dir(), "segment_profile_test.json")
        # noinspection PyBroadException
        try:
            with open(profile_path, 'r') as ff:
                data = json.load(ff, object_hook=part_hook)
            assert check_loaded_dict(data)
        except Exception:
            pytest.fail("Fail in loading profile")

    def test_measure_load(self):
        profile_path = os.path.join(get_test_dir(), "measurements_profile_test.json")
        # noinspection PyBroadException
        try:
            with open(profile_path, 'r') as ff:
                data = json.load(ff, object_hook=part_hook)
            assert check_loaded_dict(data)
        except Exception:
            pytest.fail("Fail in loading profile")

    def test_json_dump(self):
        with pytest.raises(TypeError):
            json.dumps(GaussType.Layer)
        data_string = json.dumps(GaussType.Layer, cls=PartEncoder)
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


class TestSegmentationMask:
    def test_load_seg(self):
        test_dir = get_test_dir()
        seg = LoadSegmentation.load([os.path.join(test_dir, "test_nucleus.seg")])
        assert isinstance(seg.image, str)
        assert seg.list_of_components == [1, 3]
        assert os.path.exists(os.path.join(test_dir, seg.image))

    def test_load_seg_with_image(self):
        test_dir = get_test_dir()
        seg = LoadSegmentationImage.load([os.path.join(test_dir, "test_nucleus.seg")],
                                         metadata={"default_spacing": (1, 1, 1)})
        assert isinstance(seg.image, Image)
        assert seg.list_of_components == [1, 3]
        assert isinstance(seg.segmentation, np.ndarray)
        seg.image.fit_array_to_image(seg.segmentation)

    def test_save_segmentation(self):
        test_dir = get_test_dir()
        seg = LoadSegmentationImage.load([os.path.join(test_dir, "test_nucleus.seg")],
                                         metadata={"default_spacing": (1, 1, 1)})
        assert tmp_dir != ""
        SaveSegmentation.save(os.path.join(tmp_dir, "segmentation.seg"), seg, {"relative_path": False})
        assert os.path.exists(os.path.join(tmp_dir, "segmentation.seg"))
        os.makedirs(os.path.join(tmp_dir, "seg_save"))
        save_components(seg.image, seg.list_of_components, seg.segmentation, os.path.join(tmp_dir, "seg_save"))
        assert os.path.isdir(os.path.join(tmp_dir, "seg_save"))
        assert len(glob(os.path.join(tmp_dir, "seg_save", "*"))) == 4
