from PartSegCore.image_operations import RadiusType
from PartSegCore.mask_create import MaskProperty


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


# TODO test other function from this module
