from tifffile import imread
from os import path
from partseg_utils.global_settings import src_file_folder

from partseg_utils.color_image.color_image_base import color_image, color_maps
from partseg_utils.image_operations import normalize_shape

names = list(color_maps.keys())
pos = [2, 5, 8, 12]

image = normalize_shape(imread(path.join(src_file_folder, "test_data", "A_deconv_tif.tif")))
print(image.shape)

choose_names = [names[x] for x in pos]
res_image = color_image(image[10], choose_names)