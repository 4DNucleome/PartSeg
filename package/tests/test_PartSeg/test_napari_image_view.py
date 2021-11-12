import numpy as np
from napari.layers import Image as NapariImage

from PartSeg.common_gui.napari_image_view import ImageInfo
from PartSegImage import Image


def test_image_info():
    image_info = ImageInfo(Image(np.zeros((10, 10)), image_spacing=(1, 1), axes_order="XY"), [])
    assert not image_info.coords_in([1, 1])
    assert np.all(image_info.translated_coords([1, 1]) == [1, 1])

    image_info.layers.append(NapariImage(image_info.image.get_channel(0), scale=(1, 1, 10, 10)))
    assert image_info.coords_in([0.5, 0.5, 1, 1])
    assert np.all(image_info.translated_coords([1, 1, 1, 1]) == [1, 1, 1, 1])
