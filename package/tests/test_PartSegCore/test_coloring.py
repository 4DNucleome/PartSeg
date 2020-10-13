import numpy as np
import pytest

from PartSegCore.color_image import Color, ColorMap, ColorPosition, calculate_borders, color_image_fun, create_color_map
from PartSegCore.color_image.base_colors import inferno


def arrays_are_close(arr1, arr2, num):
    return arr1.shape == arr2.shape and np.sum(arr1 != arr2) <= num


class TestCreateColorMap:
    def test_no_color(self):
        res = create_color_map(ColorMap([]))
        assert res.shape == (1024, 3)
        assert np.all(res == 0)

    def test_one_color(self):
        res = create_color_map(ColorMap([ColorPosition(0, Color(70, 50, 30))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 0] == 70)
        assert np.all(res[:, 1] == 50)
        assert np.all(res[:, 2] == 30)

        res = create_color_map(ColorMap([ColorPosition(0.5, Color(70, 50, 30))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 0] == 70)
        assert np.all(res[:, 1] == 50)
        assert np.all(res[:, 2] == 30)

        res = create_color_map(ColorMap([ColorPosition(1, Color(70, 50, 30))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 0] == 70)
        assert np.all(res[:, 1] == 50)
        assert np.all(res[:, 2] == 30)

    def test_two_colors(self):
        res = create_color_map(ColorMap([ColorPosition(0, Color(0, 0, 0)), ColorPosition(1, Color(0, 0, 0))]))
        assert res.shape == (1024, 3)
        assert np.all(res == 0)

        res = create_color_map(ColorMap([ColorPosition(0, Color(0, 0, 0)), ColorPosition(1, Color(255, 0, 0))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 1:] == 0)
        assert np.all(res[:, 0] == np.linspace(0, 256, 1024, endpoint=False, dtype=np.uint8))

        res = create_color_map(ColorMap([ColorPosition(0.25, Color(0, 0, 0)), ColorPosition(0.75, Color(255, 0, 0))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 1:] == 0)
        assert np.all(res[:255] == 0)
        assert np.all(res[-256:, 0] == 255)
        assert np.all(np.sort(res[256:-256, 0]) == res[256:-256, 0])
        assert np.sum(np.bincount(res[256:-256, 0]) != 2) < 2  # error toleration

        res = create_color_map(ColorMap([ColorPosition(0, Color(0, 255, 0)), ColorPosition(1, Color(255, 0, 0))]))
        assert res.shape == (1024, 3)
        assert np.all(res[:, 2] == 0)
        assert np.all(np.sort(res[:, 0]) == res[:, 0])
        assert np.sum(np.bincount(res[:, 0]) != 4) < 2
        assert np.all(np.sort(res[:, 1])[::-1] == res[:, 1])
        assert np.sum(np.bincount(res[:, 1]) != 4) < 4

    def test_three_colors(self):
        res = create_color_map(
            ColorMap(
                [ColorPosition(0, Color(0, 0, 0)), ColorPosition(0.5, Color(0, 0, 0)), ColorPosition(1, Color(0, 0, 0))]
            )
        )
        assert res.shape == (1024, 3)
        assert np.all(res == 0)
        res = create_color_map(
            ColorMap(
                [
                    ColorPosition(0, Color(0, 0, 0)),
                    ColorPosition(0.5, Color(255, 0, 0)),
                    ColorPosition(1, Color(0, 0, 0)),
                ]
            )
        )
        assert res.shape == (1024, 3)
        assert np.all(res[:, 1:] == 0)
        assert np.all(np.sort(res[:512, 0]) == res[:512, 0])
        assert np.sum(np.bincount(res[:512, 0]) != 2) < 1
        assert np.all(np.sort(res[512:, 0])[::-1] == res[512:, 0])
        assert np.sum(np.bincount(res[512:, 0]) != 2) < 2

    def test_three_colors_power(self):
        res = create_color_map(
            ColorMap(
                (
                    ColorPosition(0, Color(0, 0, 0)),
                    ColorPosition(0.5, Color(255, 0, 0)),
                    ColorPosition(1, Color(0, 0, 0)),
                )
            ),
            2,
        )
        assert res.shape == (1024, 3)
        assert np.all(res[:, 1:] == 0)
        assert np.all(np.sort(res[:256, 0]) == res[:256, 0])
        assert np.sum(np.bincount(res[:256, 0]) != 1) < 2
        assert np.all(np.sort(res[256:, 0])[::-1] == res[256:, 0])
        assert np.sum(np.bincount(res[256:, 0]) != 3) < 2

        res = create_color_map(
            ColorMap(
                (
                    ColorPosition(0, Color(0, 0, 0)),
                    ColorPosition(0.25, Color(255, 0, 0)),
                    ColorPosition(1, Color(0, 0, 0)),
                )
            ),
            0.5,
        )
        assert res.shape == (1024, 3)
        assert np.all(res[:, 1:] == 0)
        assert np.all(np.sort(res[:512, 0]) == res[:512, 0])
        assert np.sum(np.bincount(res[:512, 0]) != 2) < 2
        assert np.all(np.sort(res[512:, 0])[::-1] == res[512:, 0])
        assert np.sum(np.bincount(res[512:, 0]) != 2) < 2

    def test_use_color_image(self):
        array = create_color_map(ColorMap([ColorPosition(0, Color(0, 255, 0)), ColorPosition(1, Color(255, 0, 0))]))
        img = color_image_fun(
            np.linspace(0, 256, 512, endpoint=False, dtype=np.uint8).reshape((1, 512, 1)), [array], [(0, 255)]
        )
        assert img.shape == (1, 512, 3)
        assert np.all(img[0, :, 2] == 0)
        assert np.all(np.sort(img[0, :, 0]) == img[0, :, 0])
        assert np.sum(np.bincount(img[0, :, 0]) != 2) < 1


class TestArrayColorMap:
    def test_base(self):
        res = create_color_map(inferno)
        assert res.shape == (1024, 3)


class TestCalculateBorders:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_simple_2d(self, dtype):
        data = np.zeros((1, 10, 10, 10), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 0, True)
        data[:, :, 3:-3, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_sizes_2d(self, dtype):
        data = np.zeros((1, 10, 20, 30), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 0, True)
        data[0, :, 3:-3, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("num", list(range(2, 7)))
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_labels_2d(self, num, dtype):
        data = np.zeros((1, 10, num * 10, 10), dtype=dtype)
        for ind in range(num):
            data[0, 2:-2, ind * 10 + 2 : ind * 10 + 8, 2:-2] = ind + 1
        res = calculate_borders(data, 0, True)
        for ind in range(num):
            data[0, :, ind * 10 + 3 : ind * 10 + 7, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_simple_thick_2d(self, dtype):
        data = np.zeros((1, 10, 10, 10), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 1, True)
        data2 = np.copy(data)
        res2 = np.copy(res)
        res2[data == 0] = 0
        data2[0, :, 4:-4, 4:-4] = 0
        assert np.all(res2 == data2)
        assert res.dtype == dtype
        data[0, 2:-2, 1, 2:-2] = 1
        data[0, 2:-2, -2, 2:-2] = 1
        data[0, 2:-2, 2:-2, 1] = 1
        data[0, 2:-2, 2:-2, -2] = 1
        data[0, :, 4:-4, 4:-4] = 0
        assert np.all(res == data)

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_simple_3d(self, dtype):
        data = np.zeros((1, 10, 10, 10), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 0, False)
        data[0, 3:-3, 3:-3, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_sizes_3d(self, dtype):
        data = np.zeros((1, 10, 20, 30), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 0, False)
        data[0, 3:-3, 3:-3, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("num", list(range(2, 7)))
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_labels_3d(self, num, dtype):
        data = np.zeros((1, 10, num * 10, 10), dtype=dtype)
        for ind in range(num):
            data[0, 2:-2, ind * 10 + 2 : ind * 10 + 8, 2:-2] = ind + 1
        res = calculate_borders(data, 0, False)
        for ind in range(num):
            data[0, 3:-3, ind * 10 + 3 : ind * 10 + 7, 3:-3] = 0
        assert np.all(res == data)
        assert res.dtype == dtype

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_simple_thick_3d(self, dtype):
        data = np.zeros((1, 10, 10, 10), dtype=dtype)
        data[0, 2:-2, 2:-2, 2:-2] = 1
        res = calculate_borders(data, 1, False)
        data2 = np.copy(data)
        res2 = np.copy(res)
        res2[data == 0] = 0
        data2[0, 4:-4, 4:-4, 4:-4] = 0
        assert np.all(res2 == data2)
        assert res.dtype == dtype
        data[0, 2:-2, 1, 2:-2] = 1
        data[0, 2:-2, -2, 2:-2] = 1
        data[0, 2:-2, 2:-2, 1] = 1
        data[0, 2:-2, 2:-2, -2] = 1
        data[0, 1, 2:-2, 2:-2] = 1
        data[0, -2, 2:-2, 2:-2] = 1
        data[0, 4:-4, 4:-4, 4:-4] = 0
        assert np.all(res == data)
