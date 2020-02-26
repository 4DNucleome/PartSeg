import numpy as np

import pytest

from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty, fill_2d_holes_in_mask, fill_holes_in_mask, calculate_mask


class TestMaskProperty:
    def test_compare(self):
        ob1 = MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True)
        ob2 = MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True, False)
        ob3 = MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True, True)
        assert ob1 == ob2
        assert ob1 != ob3
        assert ob2 != ob3

    def test_str(self):
        text = str(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True))
        assert "dilate radius" not in text
        assert "max holes size" not in text
        text = str(MaskProperty(RadiusType.R2D, 0, RadiusType.NO, -1, False, True))
        assert "dilate radius" in text
        assert "max holes size" not in text
        text = str(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, False, True))
        assert "dilate radius" not in text
        assert "max holes size" in text
        text = str(MaskProperty(RadiusType.R3D, 0, RadiusType.R2D, -1, False, True))
        assert "dilate radius" in text
        assert "max holes size" in text


class TestFillHoles:
    def test_fill_2d(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[:, 4:6, 4:6] = 0
        assert np.all(mask == fill_2d_holes_in_mask(mask2, -1))

    def test_fill_2d_min_size(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[:, 4:6, 4:6] = 0
        mask3 = fill_2d_holes_in_mask(mask2, 1)
        assert np.all(mask3 == mask2)
        mask3 = fill_2d_holes_in_mask(mask2, 3)
        assert np.all(mask3 == mask2)
        mask3 = fill_2d_holes_in_mask(mask2, 4)
        assert np.all(mask3 == mask)
        mask3 = fill_2d_holes_in_mask(mask2, 15)
        assert np.all(mask3 == mask)

    def test_fill_3d(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[:, 4:6, 4:6] = 0
        assert not np.all(mask == fill_holes_in_mask(mask2, -1))
        mask2 = np.copy(mask)
        mask2[4:6, 4:6, 4:6] = 0
        assert np.all(mask == fill_holes_in_mask(mask2, -1))

    def test_fill_3d_min_size(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[4:6, 4:6, 4:6] = 0
        assert not np.all(mask == fill_holes_in_mask(mask2, 6))
        assert np.all(mask == fill_holes_in_mask(mask2, 8))
        assert np.all(mask == fill_holes_in_mask(mask2, 16))

    def test_on_2d_data(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[4:6, 4:6] = 0
        assert np.all(mask == fill_holes_in_mask(mask2, -1))
        assert np.all(mask == fill_2d_holes_in_mask(mask2, -1))


class TestCalculateMask:
    def test_single(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        assert np.all(mask2 == mask)
        mask3 = np.copy(mask)
        mask3[mask > 0] = 2
        mask2 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        assert np.all(mask2 == mask)

    def test_dilate(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = calculate_mask(MaskProperty(RadiusType.R3D, -1, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        mask3 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask3[3:7, 3:7, 3:7] = 1
        assert np.all(mask2 == mask3)
        mask2 = calculate_mask(MaskProperty(RadiusType.R2D, -1, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        mask3 = np.zeros((10, 10, 10), dtype=np.uint8)
        mask3[2:8, 3:7, 3:7] = 1
        assert np.all(mask2 == mask3)

    def test_fil_holes_3d(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[4:6, 4:6, 4:6] = 0
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, 7, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask2)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, 8, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)

    def test_fil_holes_3d_torus(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask[:, 4:6, 4:6] = 0
        mask2 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, False, True), mask, None, (1, 1, 1))
        assert np.all(mask2 == mask)

    def test_fil_holes_2d(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[:, 4:6, 4:6] = 0
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, 3, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask2)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, 4, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)

    def test_save_components(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[2:8, 4:6, 4:6] = 2
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask2)
        mask4 = np.copy(mask2)
        mask4[mask4 == 2] = 3
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, True, True), mask4, None, (1, 1, 1))
        assert np.all(mask3 == mask4)

    def test_save_component_fill_holes(self):
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[2:7, 2:-2, 2:-2] = 1
        mask[7:-2, 2:-2, 2:-2] = 2
        mask[4:-4, 4:-4, 4:-4] = 3
        mask2 = np.copy(mask)
        mask2[5, 5, 5] = 0
        mask2[3, 3, 3] = 0
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask[2:-2, 4:-4, 4:-4] = 3
        mask2 = np.copy(mask)
        mask2[5, 5, 5] = 0
        mask2[3, 3, 3] = 0
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask[2:7, 2:-2, 2:-2] = 1
        mask[4:-4, 4:-4, 4:-4] = 2
        mask2 = np.copy(mask)
        mask2[5, 5, 5] = 0
        mask2[3, 3, 3] = 0
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)

    @pytest.mark.xfail(reason="problem with alone pixels")
    def test_save_component_fill_holes_problematic(self):
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[2:7, 2:-2, 2:-2] = 3
        mask[7:-2, 2:-2, 2:-2] = 2
        mask[4:-4, 4:-4, 4:-4] = 1
        mask2 = np.copy(mask)
        mask2[5, 5, 5] = 0
        mask2[3, 3, 3] = 0
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)
        mask1 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, True, True), mask2, None, (1, 1, 1))
        assert np.all(mask == mask1)

    def test_reverse(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:7, 3:7, 3:7] = 1
        mask2 = np.ones((10, 10, 10), dtype=np.uint8)
        mask2[mask > 0] = 0
        mask3 = calculate_mask(
            MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True, True), mask, None, (1, 1, 1)
        )
        assert np.all(mask3 == mask2)

    def test_clip(self):
        mask = np.zeros((10, 10, 10), dtype=np.uint8)
        mask[3:7, 3:7, 3:7] = 1
        mask2 = calculate_mask(MaskProperty(RadiusType.R3D, 1, RadiusType.NO, -1, False, True), mask, mask, (1, 1, 1))
        assert np.all(mask == mask2)
        mask2[:, 4:6, 4:6] = 0
        mask3 = calculate_mask(MaskProperty(RadiusType.R3D, 1, RadiusType.NO, -1, False, True), mask, mask2, (1, 1, 1))
        assert np.all(mask3 == mask2)
        mask2[:] = 0
        mask2[4:6, 4:6, 4:6] = 1
        mask3 = calculate_mask(
            MaskProperty(RadiusType.NO, 0, RadiusType.NO, -1, False, True, True), mask2, mask, (1, 1, 1)
        )
        mask[mask2 == 1] = 0
        assert np.all(mask3 == mask)

    def test_chose_components(self):
        mask = np.zeros((12, 12, 12), dtype=np.uint8)
        mask[2:7, 2:-2, 2:-2] = 3
        mask[7:-2, 2:-2, 2:-2] = 2
        mask[4:-4, 4:-4, 4:-4] = 1
        mask1 = calculate_mask(
            MaskProperty(RadiusType.NO, 0, RadiusType.R2D, -1, True, True), mask, None, (1, 1, 1), [1, 3]
        )
        assert np.all(np.unique(mask1) == [0, 1, 3])


# TODO add test with touching boundaries.
