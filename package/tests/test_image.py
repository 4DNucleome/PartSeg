import numpy as np

from PartSegImage import Image


class TestImage:
    def test_fit_mask_simple(self):
        data = np.zeros((1, 10, 20, 20, 1), np.uint8)
        image = Image(data, (1, 1, 1), "")
        mask = np.zeros((1, 10, 20, 20), np.uint8)
        mask[0, 2:-2, 4:-4, 4:-4] = 5
        image.fit_mask_to_image(mask)

    def test_fit_mask_mapping_val(self):
        data = np.zeros((1, 10, 20, 20, 1), np.uint8)
        image = Image(data, (1, 1, 1), "")
        mask = np.zeros((1, 10, 20, 20), np.uint16)
        mask[0, 2:-2, 4:-4, 4:10] = 5
        mask[0, 2:-2, 4:-4, 11:-4] = 7
        mask2 = image.fit_mask_to_image(mask)
        assert np.all(np.unique(mask2) == [0, 1, 2])
        assert np.all(np.unique(mask) == [0, 5, 7])
        map_arr = np.array([0, 0, 0, 0, 0, 1, 0, 2])
        assert np.all(map_arr[mask] == mask2)
        assert mask2.dtype == np.uint8
