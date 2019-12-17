import numpy as np
import pytest

from PartSeg.common_backend.base_settings import LabelColorDict


class TestLabelColorDict:
    def test_has_default(self):
        dkt = LabelColorDict({})
        assert "default" in dkt
        assert isinstance(dkt.get_array("default"), np.ndarray)
        assert dkt.get_array("default").dtype == np.uint8
        assert isinstance(dkt["default"][1], bool)
        assert dkt["default"][1] is False

    def test_delete_default(self):
        dkt = LabelColorDict({})
        with pytest.raises(ValueError):
            del dkt["default"]
        assert "default" in dkt

    def test_add_element(self):
        dkt = LabelColorDict({})
        labels = [[1, 2, 3], [120, 230, 100]]
        with pytest.raises(ValueError):
            dkt["test"] = labels
        assert "test" not in dkt
        dkt["custom_test"] = labels
        assert "custom_test" in dkt
        assert dkt["custom_test"][1] is True
        assert isinstance(dkt.get_array("custom_test"), np.ndarray)
        assert dkt.get_array("custom_test").dtype == np.uint8
        del dkt["custom_test"]
        assert "custom_test" not in dkt
