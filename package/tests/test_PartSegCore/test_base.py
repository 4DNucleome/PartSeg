import numpy as np
import pytest

from PartSegCore.image_operations import gaussian, median


class TestImageOperation:
    @pytest.mark.parametrize("dims", [2, 3, 4])
    @pytest.mark.parametrize("method", [gaussian, median])
    @pytest.mark.parametrize("per_layer", [True, False])
    def test_filter_dims_dummy(self, dims, method, per_layer):
        data = np.zeros((2,) * (dims - 2) + (5, 5), dtype=np.uint8)
        res = method(data, 1, per_layer)
        assert np.all(res == 0)

    @pytest.mark.parametrize("dims", [2, 3, 4])
    @pytest.mark.parametrize("method", [gaussian, median])
    @pytest.mark.parametrize("per_layer", [True, False])
    def test_filter_dims_ones(self, dims, method, per_layer):
        data = np.zeros((2,) * (dims - 2) + (5, 5), dtype=np.uint8)
        slices = (0,) * (dims - 2) + (1,)
        data[slices] = 1
        res = method(data, 2, per_layer)
        assert not np.all(res == data)
