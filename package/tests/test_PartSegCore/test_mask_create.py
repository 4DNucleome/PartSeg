import numpy as np
import pytest

from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty, calculate_mask, fill_2d_holes_in_mask, fill_holes_in_mask
from PartSegImage import Image


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

    def test_simple_mask(self):
        assert MaskProperty.simple_mask() == MaskProperty(RadiusType.NO, 0, RadiusType.NO, 0, False, False)


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
        mask = np.zeros((1, 10, 10, 10), dtype=np.uint8)
        mask[0, 2:8, 2:8, 2:8] = 1
        mask2 = calculate_mask(MaskProperty(RadiusType.R3D, -1, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        mask3 = np.zeros((1, 10, 10, 10), dtype=np.uint8)
        mask3[0, 3:7, 3:7, 3:7] = 1
        assert np.all(mask2 == mask3)
        mask2 = calculate_mask(MaskProperty(RadiusType.R2D, -1, RadiusType.NO, -1, False, True), mask, None, (1, 1, 1))
        mask3 = np.zeros((1, 10, 10, 10), dtype=np.uint8)
        mask3[0, 2:8, 3:7, 3:7] = 1
        assert np.all(mask2 == mask3)

    def test_fil_holes_3d(self):
        mask = np.zeros((1, 10, 10, 10), dtype=np.uint8)
        mask[0, 2:8, 2:8, 2:8] = 1
        mask2 = np.copy(mask)
        mask2[0, 4:6, 4:6, 4:6] = 0
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, -1, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, 7, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask2)
        mask3 = calculate_mask(MaskProperty(RadiusType.NO, 0, RadiusType.R3D, 8, False, True), mask2, None, (1, 1, 1))
        assert np.all(mask3 == mask)

    def test_fil_holes_3d_torus(self):
        mask = np.zeros((1, 10, 10, 10), dtype=np.uint8)
        mask[0, 2:8, 2:8, 2:8] = 1
        mask[0, :, 4:6, 4:6] = 0
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

    @pytest.mark.parametrize("dilate", [RadiusType.NO, RadiusType.R2D, RadiusType.R3D])
    @pytest.mark.parametrize("radius", [-1, 0, 1, 3])
    @pytest.mark.parametrize("fill_holes", [RadiusType.NO, RadiusType.R2D, RadiusType.R3D])
    @pytest.mark.parametrize("max_holes_size", [-1, 0, 100])
    @pytest.mark.parametrize("save_components", [True, False])
    @pytest.mark.parametrize("clip_to_mask", [True, False])
    @pytest.mark.parametrize("reversed_mask", [True, False])
    @pytest.mark.parametrize("old_mask", [True, False])
    def test_mask_property_combinations(
        self, dilate, radius, fill_holes, max_holes_size, save_components, clip_to_mask, reversed_mask, old_mask
    ):
        mask = np.zeros((1, 6, 6, 15), dtype=np.uint8)
        im = Image(data=mask.copy(), image_spacing=(3, 1, 1), file_path="", axes_order="TZYX")
        mask[:, 1:-1, 1:-1, 2:5] = 1
        mask[:, 2:-2, 2:-2, 3:4] = 0
        mask[:, 1:-1, 1:-1, 6:9] = 2
        mask[:, 2:-2, 2:-2, 7:8] = 0
        mask[:, 1:-1, 1:-1, 10:13] = 3
        mask[:, 2:-2, 2:-2, 11:12] = 0
        mask = im.fit_mask_to_image(mask)
        assert np.all(np.unique(mask.flat) == [0, 1, 2, 3])
        _old_mask = np.zeros(mask.shape, dtype=mask.dtype) if old_mask else None
        mp = MaskProperty(
            dilate=dilate,
            dilate_radius=radius,
            fill_holes=fill_holes,
            max_holes_size=max_holes_size,
            save_components=save_components,
            clip_to_mask=clip_to_mask,
            reversed_mask=reversed_mask,
        )
        mask1 = calculate_mask(mp, mask, _old_mask, im.spacing, time_axis=im.time_pos)
        assert mask1.shape == mask.shape

    @pytest.mark.parametrize("time", [1, 2, 4])
    def test_time_axis(self, time):
        mask = np.zeros((time, 6, 6, 6), dtype=np.uint8)
        mask[:, 1:-1, 1:-1, 2:5] = 1
        mask[:, 2:-2, 2:-2, 3:4] = 0
        mp = MaskProperty(
            dilate=RadiusType.NO,
            dilate_radius=0,
            fill_holes=RadiusType.NO,
            max_holes_size=-1,
            save_components=False,
            clip_to_mask=False,
        )
        mask1 = calculate_mask(mp, mask, None, (1, 1, 1))
        assert mask1.shape == mask.shape


# TODO add test with touching boundaries.
