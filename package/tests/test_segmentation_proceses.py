import os.path
import json
import pytest

from PartSeg.tiff_image import ImageReader
from PartSeg.utils.analysis.algorithm_description import analysis_algorithm_dict, SegmentationProfile
from PartSeg.utils.analysis.save_hooks import part_hook
from PartSeg.utils.json_hooks import check_loaded_dict
from PartSeg.utils.segmentation.algorithm_base import SegmentationAlgorithm


def empty(_a, _b):
    pass


class TestSegmentation:
    @staticmethod
    def get_test_dir():
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")

    def test_profile_execute(self):
        profile_path = os.path.join(self.get_test_dir(), "segment_profile_test.json")
        # noinspection PyBroadException
        try:
            with open(profile_path, 'r') as ff:
                data = json.load(ff, object_hook=part_hook)
            assert check_loaded_dict(data)
        except Exception:
            pytest.fail("Fail in loading profile")
            return
        image = ImageReader.read_image(
            os.path.join(self.get_test_dir(), "stack1_components", "stack1_component5.tif"),
            os.path.join(self.get_test_dir(), "stack1_components", "stack1_component5_mask.tif")
        )

        val: SegmentationProfile
        for val in data.values():
            algorithm: SegmentationAlgorithm = analysis_algorithm_dict[val.algorithm]()
            algorithm.set_image(image)
            algorithm.set_mask(image.mask.squeeze())
            algorithm.set_parameters(**val.values)
            result = algorithm.calculation_run(empty)
            assert result.segmentation.max() == 2
