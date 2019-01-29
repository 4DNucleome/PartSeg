from PartSeg.tiff_image import ImageReader, Image
from PartSeg.utils.global_settings import static_file_folder
import os.path

class TestImageClass():
    def test_image_read(self):
        image = ImageReader.read_image(os.path.join(static_file_folder, "initial_images", "stack.tif"))
        assert isinstance(image, Image)