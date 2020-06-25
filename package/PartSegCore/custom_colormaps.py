import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "black_red",
]

red_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
}

blue_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
}

green_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "blue": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
}

magenta_dict = {
    "red": ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
    "green": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    "blue": ((0.0, 0.0, 0.0), (1.0, 144.0 / 255, 144.0 / 255)),
}

black_red = LinearSegmentedColormap("BlackRed", red_dict)
plt.register_cmap(cmap=black_red)

black_green = LinearSegmentedColormap("BlackGreen", green_dict)
plt.register_cmap(cmap=black_green)

black_blue = LinearSegmentedColormap("BlackBlue", blue_dict)
plt.register_cmap(cmap=black_blue)

black_magenta = LinearSegmentedColormap("BlackMagenta", magenta_dict)
plt.register_cmap(cmap=black_magenta)
