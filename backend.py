from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot


class Profile:
    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        """
        self.name = name
        self.threshold = threshold
        self.threshold_list = threshold_list
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size


class Settings(object):
    """
    :type profiles: dict[str, Profile]
    :type threshold: int
    :type threshold_list: list[int]
    :type threshold_type: str
    :type minimum_size: int
    :type image: np.ndarray
    :type image_change_callback_: list[() -> None]
    """
    def __init__(self, setings_path):
        # TODO Reading setings from file
        self.color_map_name = "cubehelix"
        self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
        self.callback_color_map = []
        self.profiles = {}
        self.threshold = 33000
        self.threshold_list = []
        self.threshold_type = "Upper"
        self.threshold_layer_separate = False
        self.minimum_size = 100
        self.image = None
        self.image_change_callback_ = []

    def change_colormap(self, new_color_map):
        """
        :type new_color_map: str
        :param new_color_map: name of new colormap
        :return:
        """
        self.color_map_name = new_color_map
        self.color_map = matplotlib.cm.get_cmap(new_color_map)
        for fun in self.callback_color_map:
            print("buka")
            fun()

    def add_colormap_callback(self, callback):
        self.callback_color_map.append(callback)

    def remove_colormap_callback(self, callback):
        self.callback_color_map.remove(callback)

    def create_new_profile(self, name, overwrite=False):
        """
        :type name: str
        :type overwrite: bool
        :param name: Profile name
        :param overwrite: Overwrite existing profile
        :return:
        """
        if not overwrite and name in self.profiles:
            raise ValueError("Profile with this name already exists")
        self.profiles[name] = Profile(name, self.threshold, self.threshold_list, self.threshold_type, self.minimum_size)

    @property
    def colormap_list(self):
        return pyplot.colormaps()


class Segment(object):
    def __init__(self):
        pass
