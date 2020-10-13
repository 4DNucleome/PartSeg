import numpy as np

from PartSegCore import UNIT_SCALE, Units
from PartSegCore.mask_partition_utils import BorderRim, MaskDistanceSplit


class TestBorderRim:
    def test_base_2d(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:-1, 1:-1] = 1
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (1 / nm_scalar,) * 2
        result_mask = BorderRim.border_mask(mask, 1, Units.nm, voxel_size)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2] = 0
        assert np.all(mask2 == result_mask)

    def test_scaling_2d(self):
        mask = np.zeros((6, 12), dtype=np.uint8)
        mask[1:-1, 2:-2] = 1
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (2 / nm_scalar, 1 / nm_scalar)
        result_mask = BorderRim.border_mask(mask, 2, Units.nm, voxel_size)
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4] = 0
        assert np.all(mask2 == result_mask)

    def test_base_3d(self):
        mask = np.zeros((20, 20, 20), dtype=np.uint8)
        mask[2:-2, 2:-2, 2:-2] = 1
        mask2 = np.copy(mask)
        mask2[3:-3, 3:-3, 3:-3] = 0
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (1 / nm_scalar,) * 3
        result_mask = BorderRim.border_mask(mask, 1, Units.nm, voxel_size)
        assert np.all(result_mask == mask2)

    def test_scaling_3d(self):
        mask = np.zeros((10, 20, 20), dtype=np.uint8)
        mask[1:-1, 2:-2, 2:-2] = 1
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4, 4:-4] = 0
        nm_scalar = UNIT_SCALE[Units.nm.value]
        voxel_size = (2 / nm_scalar, 1 / nm_scalar, 1 / nm_scalar)
        result_mask = BorderRim.border_mask(mask, 2, Units.nm, voxel_size)
        assert np.all(result_mask == mask2)


class TestSplitMaskOnPart:
    def test_base_2d_thick(self):
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:-1, 1:-1] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, False, (1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, False, (1, 1))
        assert np.all(mask2 == result_mask)

    def test_scaling_2d_thick(self):
        mask = np.zeros((6, 12), dtype=np.uint8)
        mask[1:-1, 2:-2] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, False, (2, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, False, (2, 1))
        assert np.all(mask2 == result_mask)

    def test_base_3d_thick(self):
        mask = np.zeros((6, 6, 6), dtype=np.uint8)
        mask[1:-1, 1:-1, 1:-1] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, False, (1, 1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2, 2:-2] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, False, (1, 1, 1))
        assert np.all(mask2 == result_mask)

    def test_scaling_3d_thick(self):
        mask = np.zeros((6, 12, 12), dtype=np.uint8)
        mask[1:-1, 2:-2, 2:-2] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, False, (2, 1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4, 4:-4] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, False, (2, 1, 1))
        assert np.all(mask2 == result_mask)

    def test_base_2d_volume(self):
        mask = np.zeros((7, 7), dtype=np.uint8)
        mask[1:-1, 1:-1] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, True, (1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, True, (1, 1))
        assert np.all(mask2 == result_mask)

    def test_scaling_2d_volume(self):
        mask = np.zeros((7, 14), dtype=np.uint8)
        mask[1:-1, 2:-2] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, True, (2, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, True, (2, 1))
        assert np.all(mask2 == result_mask)

    def test_base_3d_volume(self):
        mask = np.zeros((7, 7, 7), dtype=np.uint8)
        mask[1:-1, 1:-1, 1:-1] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, True, (1, 1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 2:-2, 2:-2] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, True, (1, 1, 1))
        assert np.all(mask2 == result_mask)

    def test_scaling_3d_volume(self):
        mask = np.zeros((7, 14, 14), dtype=np.uint8)
        mask[1:-1, 2:-2, 2:-2] = 1
        result_mask = MaskDistanceSplit.split(mask, 1, True, (2, 1, 1))
        assert np.all(mask == result_mask)
        mask2 = np.copy(mask)
        mask2[2:-2, 4:-4, 4:-4] = 2
        result_mask = MaskDistanceSplit.split(mask, 2, True, (2, 1, 1))
        assert np.all(mask2 == result_mask)
