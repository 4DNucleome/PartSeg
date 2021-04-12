import numpy as np

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.project_info import AdditionalLayerDescription
from PartSegCore.segmentation.algorithm_base import SegmentationResult


class TestSegmentationResult:
    def test_update_alternative_names(self):
        res = SegmentationResult(
            roi=np.zeros((10, 10), dtype=np.uint8),
            parameters=ROIExtractionProfile("test", "test", {}),
            additional_layers={
                "test1": AdditionalLayerDescription(np.zeros((10, 10), dtype=np.uint8), "image", "aa"),
                "test2": AdditionalLayerDescription(np.zeros((10, 10), dtype=np.uint8), "image", ""),
            },
        )
        assert res.additional_layers["test1"].name == "aa"
        assert res.additional_layers["test2"].name == "test2"
