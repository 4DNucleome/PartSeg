from partseg_utils.custom_colormaps import black_blue
from partseg_utils.colors import default_colors
from matplotlib import pyplot
from matplotlib.cm import get_cmap
import numpy as np

print(black_blue, default_colors)
if __name__ == '__main__':
    result = dict()
    base_array = np.linspace(0, 1, 1024)
    for colormap in pyplot.colormaps():
        color_array = np.copy(base_array)
        colored = get_cmap(colormap)(color_array)
        colored = (colored * 255).astype(np.uint8)
        result[colormap] = colored[..., :-1]
    np.savez("static_files/colors.npz", **result)


