from PartSegCore_compiled_backend.color_image_cython import add_labels, calculate_borders, color_grayscale, resolution

from .base_colors import default_colormap_dict, default_label_dict

__all__ = (
    "add_labels",
    "calculate_borders",
    "color_grayscale",
    "default_colormap_dict",
    "default_label_dict",
    "resolution",
)
