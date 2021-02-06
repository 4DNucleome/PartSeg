from PartSegCore_compiled_backend.color_image_cython import add_labels, calculate_borders, color_grayscale, resolution

from .base_colors import default_colormap_dict, default_label_dict

# from .color_image_base import color_image_fun, create_color_map

__all__ = (
    "add_labels",
    "calculate_borders",
    "color_grayscale",
    "default_colormap_dict",
    "default_label_dict",
    "resolution",
)
