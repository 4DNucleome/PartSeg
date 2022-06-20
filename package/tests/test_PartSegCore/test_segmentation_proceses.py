# pylint: disable=R0201

import os.path

import numpy as np
import pytest

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.io_utils import load_metadata_base
from PartSegCore.segmentation.algorithm_base import ROIExtractionAlgorithm
from PartSegCore.utils import check_loaded_dict
from PartSegImage import TiffImageReader


def empty(_a, _b):
    pass  # pragma: no cover


class TestSegmentation:
    def test_profile_execute(self, data_test_dir):
        profile_path = os.path.join(data_test_dir, "segment_profile_test.json")
        # noinspection PyBroadException
        try:
            data = load_metadata_base(profile_path)
            assert check_loaded_dict(data)
        except Exception:  # pylint: disable=W0703  # pragma: no cover
            pytest.fail("Fail in loading profile")
            return
        image = TiffImageReader.read_image(
            os.path.join(data_test_dir, "stack1_components", "stack1_component5.tif"),
            os.path.join(data_test_dir, "stack1_components", "stack1_component5_mask.tif"),
        )

        val: ROIExtractionProfile
        for val in data.values():
            algorithm: ROIExtractionAlgorithm = AnalysisAlgorithmSelection[val.algorithm]()
            algorithm.set_image(image)
            algorithm.set_mask(image.mask.squeeze())
            algorithm.set_parameters(val.values)
            result = algorithm.calculation_run(empty)
            assert np.max(result.roi) == 2
