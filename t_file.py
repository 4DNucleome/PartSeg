from __future__ import print_function
from qt_import import QLabel, QPixmap, QImage
import tifffile as tif

im = tif.imread("stack.tif")
print(im.shape)