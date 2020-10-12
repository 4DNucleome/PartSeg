import os.path

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.load_functions import UpdateLoadedMetadataAnalysis
from PartSegCore.json_hooks import check_loaded_dict
from PartSegCore.segmentation.algorithm_base import SegmentationAlgorithm
from PartSegImage import TiffImageReader


def empty(_a, _b):
    pass


class TestSegmentation:
    def test_profile_execute(self, data_test_dir):
        profile_path = os.path.join(data_test_dir, "segment_profile_test.json")
        # noinspection PyBroadException
        try:
            data = UpdateLoadedMetadataAnalysis.load_json_data(profile_path)
            assert check_loaded_dict(data)
        except Exception:  # pylint: disable=W0703
            pytest.fail("Fail in loading profile")
            return
        image = TiffImageReader.read_image(
            os.path.join(data_test_dir, "stack1_components", "stack1_component5.tif"),
            os.path.join(data_test_dir, "stack1_components", "stack1_component5_mask.tif"),
        )

        val: ROIExtractionProfile
        for val in data.values():
            algorithm: SegmentationAlgorithm = analysis_algorithm_dict[val.algorithm]()
            algorithm.set_image(image)
            algorithm.set_mask(image.mask.squeeze())
            algorithm.set_parameters(**val.values)
            result = algorithm.calculation_run(empty)
            assert np.max(result.roi) == 2
