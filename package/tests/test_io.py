from PartSeg.tiff_image import ImageReader, Image
from PartSeg.utils.global_settings import static_file_folder
import os.path
import numpy as np

class TestImageClass():
    def get_test_dir(self):
        return os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "test_data")

    def test_image_read(self):
        image = ImageReader.read_image(os.path.join(static_file_folder, "initial_images", "stack.tif"))
        assert isinstance(image, Image)

    def test_lsm_read(self):
        test_dir = self.get_test_dir()
        image1 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.lsm"))
        image2 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.tif"))
        data = np.load(os.path.join(test_dir, "test_lsm.npy"))
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())

    def _test_ome_read(self):  # error in tifffile
        test_dir = self.get_test_dir()
        test_dir = self.get_test_dir()
        image1 = ImageReader.read_image(os.path.join(test_dir, "test_lsm2.tif"))
        image2 = ImageReader.read_image(os.path.join(test_dir, "test_lsm.tif"))
        data = np.load(os.path.join(test_dir, "test_lsm.npy"))
        assert np.all(image1.get_data() == data)
        assert np.all(image2.get_data() == data)
        assert np.all(image1.get_data() == image2.get_data())

