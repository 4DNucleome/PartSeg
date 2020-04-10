from PartSegImage import GenericImageReader
from segmentify import Viewer, gui_qt, util

image = GenericImageReader.read_image("test_data/test_nucleus.tif")


img = util.parse_img()

with gui_qt():
    viewer = Viewer(img)
