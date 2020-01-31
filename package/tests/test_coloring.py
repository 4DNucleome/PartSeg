import numpy as np

from PartSegCore.color_image.base_colors import inferno
from PartSegCore.color_image import Color, ColorPosition, ColorMap, create_color_map, color_image_fun


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
