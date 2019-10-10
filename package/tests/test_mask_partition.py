import numpy as np

from PartSegCore import Units, UNIT_SCALE
from PartSegCore.mask_partition_utils import BorderRim, SplitMaskOnPart


class TestBorderRim:
    # TODO add 2d tests
    def test_base(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1
        mask2 = np.copy(mask)
        mask2[3:-3, 3:-3, 3:-3] = 0
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (1/nm_scalar,) * 3
        result_mask = BorderRim.border_mask(mask, 1, Units.nm, voxel_size)
        assert np.all(result_mask == mask2)

    def test_scaling(self):
        mask = np.zeros((10, 20, 20), dtype=np.uint8)
        mask[1:-1, 2:-2, 2:-2] = 1
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4, 4:-4] = 0
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (2/nm_scalar, 1/nm_scalar, 1/nm_scalar)
        result_mask = BorderRim.border_mask(mask, 2, Units.nm, voxel_size)
        assert np.all(result_mask == mask2)


class TestSplitMaskOnPart:
    # TODO Add more tests
    def test_base(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:-1, 1:-1] = 1
        result_mask = SplitMaskOnPart.split(mask, 1, False, (1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2] = 2
        result_mask = SplitMaskOnPart.split(mask, 2, False, (1, 1))
        assert np.all(mask2 == result_mask)
