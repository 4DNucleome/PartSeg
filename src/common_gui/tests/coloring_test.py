from common_gui.color_image import color_image, color_maps
from project_utils.image_operations import normalize_shape
from tifffile import imread

names = list(color_maps.keys())
pos = [2, 5, 8, 12]

image = normalize_shape(imread("../../../data/merge.tif"))
print(image.shape)

choose_names = [names[x] for x in pos]
res_image = color_image(image[10], choose_names)