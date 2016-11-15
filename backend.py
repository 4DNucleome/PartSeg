from __future__ import print_function, division
import matplotlib
from matplotlib import pyplot
import numpy as np
import SimpleITK as sitk
import h5py
import json
import tempfile
import os
import tarfile

UPPER = "Upper"
GAUSS = "Gauss"


def class_to_dict(obj, *args):
    """
    Create dict which contains values of given fields
    :type obj: object
    :type args: list[str]
    :return:
    """
    res = dict()
    for name in args:
        res[name] = getattr(obj, name)
    return res


def dict_set_class(obj, dic, *args):
    """
    Set fields of given object based on values from dict.
    If *args contains no names all values from dict are used
    :type obj: object
    :type dic: dict[str,object]
    :param args: list[str]
    :return:
    """
    if len(args) == 0:
        li = dic.keys()
    else:
        li = args
    for name in li:
        tt = getattr(obj, name)

        setattr(obj, name, dic[name])


def gaussian(image, radius):
    """
    :param image: image to apply gausian filter
    :param radius: radius for gaussian kernel
    :return:
    """
    if len(image.shape) == 2:
        return sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(image), radius))
    res = np.copy(image)
    for layer in res:
        layer[...] = sitk.GetArrayFromImage(sitk.DiscreteGaussian(sitk.GetImageFromArray(layer), radius))
    return res


def bisect(arr, val, comp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if comp(arr[e], val): l = e
        else: r = e
    return r


class Profile:
    def __init__(self, name, threshold, threshold_list, threshold_type, minimum_size, use_gauss):
        """
        :param name: str,
        :param threshold: int
        :param threshold_list: list[int]
        :param threshold_type: str
        :param minimum_size: int
        :param use_gauss: bool
        """
        self.name = name
        self.threshold = threshold
        self.threshold_list = threshold_list
        self.threshold_type = threshold_type
        self.minimum_size = minimum_size
        self.use_gauss = use_gauss

    def __str__(self):
        text = self.name + "\n"

        return text


class Settings(object):
    """
    :type profiles: dict[str, Profile]
    :type threshold: int
    :type threshold_list: list[int]
    :type threshold_type: str
    :type minimum_size: int
    :type image: np.ndarray
    :type image_change_callback: list[() -> None]
    """
    def __init__(self, setings_path):
        # TODO Reading setings from file
        self.color_map_name = "cubehelix"
        self.color_map = matplotlib.cm.get_cmap(self.color_map_name)
        self.callback_colormap = []
        self.callback_colormap_list = []
        self.callback_change_layer = []
        self.chosen_colormap = pyplot.colormaps()
        self.profiles = dict()
        self.use_gauss = False
        self.use_draw_result = False
        self.draw_callback = []
        self.threshold = 33000
        self.threshold_list = []
        self.threshold_type = UPPER
        self.threshold_layer_separate = False
        self.minimum_size = 100
        self.overlay = 0.7
        self.mask_overlay = 0.7
        self.power_norm = 1
        self.image = None
        self.gauss_image = None
        self.mask = None
        self.image_change_callback = []
        self.threshold_change_callback = []
        self.threshold_type_change_callback = []
        self.minimum_size_change_callback = []
        self.metadata_changed_callback = []
        self.layer_num = 0
        self.open_directory = None
        self.open_filter = None
        self.save_directory = None
        self.save_filter = None
        self.spacing = [5, 5, 30]
        self.voxel_size = [70, 70, 210]
        self.size_unit = "nm"
        self.advanced_menu_geometry = None
        self.file_path = ""
        self.protect = False
        self.load(setings_path)
        self.prev_segmentation_settings = []
        self.next_segmentation_settings = []

    def dump(self, file_path):
        important_data = \
            class_to_dict(self, "open_directory", "open_filter", "save_directory", "save_filter", "spacing",
                          "voxel_size", "size_unit", "threshold", "color_map_name", "overlay", "minimum_size")
        with open(file_path, "w") as ff:
            json.dump(important_data, ff)

    def load(self, file_path):
        try:
            with open(file_path, "r") as ff:
                important_data = json.load(ff)
            dict_set_class(self, important_data, "open_directory", "open_filter", "save_directory", "save_filter",
                           "spacing", "voxel_size", "size_unit", "threshold", "color_map_name", "overlay",
                           "minimum_size")
        except IOError:
            print("No configuration file")
            pass
        except KeyError:
            print("Bad configuration")

    def change_colormap(self, new_color_map):
        """
        :type new_color_map: str
        :param new_color_map: name of new colormap
        :return:
        """
        self.color_map_name = new_color_map
        self.color_map = matplotlib.cm.get_cmap(new_color_map)
        for fun in self.callback_colormap:
            fun()

    def add_colormap_list_callback(self, callback):
        self.callback_colormap_list.append(callback)

    def add_colormap_callback(self, callback):
        self.callback_colormap.append(callback)

    def remove_colormap_callback(self, callback):
        self.callback_colormap.remove(callback)

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
        return self.chosen_colormap

    @property
    def available_colormap_list(self):
        return pyplot.colormaps()

    def add_image(self, image, file_path, mask=None,):
        self.image = image
        self.gauss_image = gaussian(self.image, 1)
        self.mask = mask
        self.file_path = file_path
        self.threshold_list = []
        self.threshold_layer_separate = False
        for fun in self.image_change_callback:
            if isinstance(fun, tuple) and fun[1] == str:
                fun[0](file_path)
                continue
            if isinstance(fun, tuple) and fun[1] == GAUSS:
                fun[0](image, self.gauss_image)
                continue
            fun(image)

    def add_image_callback(self, callback):
        self.image_change_callback.append(callback)

    def change_threshold(self, new_threshold):
        if self.protect:
            return
        if self.threshold_layer_separate:
            if self.threshold_list[self.layer_num] == new_threshold:
                return
            self.threshold_list[self.layer_num] = new_threshold
        else:
            if self.threshold == new_threshold:
                return
            self.threshold = new_threshold
        for fun in self.threshold_change_callback:
            fun()

    def add_change_layer_callback(self, callback):
        self.callback_change_layer.append(callback)

    def change_layer(self, val):
        self.layer_num = val
        self.protect = True
        if self.threshold_layer_separate:
            for fun in self.callback_change_layer:
                fun(self.threshold_list[val])
        else:
            for fun in self.callback_change_layer:
                fun(self.threshold)
        # for fun in self.threshold_change_callback:
        #     fun()
        self.protect = False

    def change_threshold_type(self, new_type):
        print(new_type)
        if new_type == "Upper threshold:":
            self.threshold_type = UPPER
        else:
            self.threshold_type = "Lower"
        for fun in self.threshold_change_callback:
            fun()
        for fun in self.threshold_type_change_callback:
            fun()

    def change_layer_threshold(self, layer_threshold):
        self.threshold_layer_separate = layer_threshold
        if layer_threshold and self.threshold_list == []:
            self.threshold_list = [self.threshold] * self.image.shape[0]
        for fun in self.threshold_change_callback:
            fun()

    def change_gauss(self, use_gauss):
        self.use_gauss = bool(use_gauss)
        for fun in self.threshold_change_callback:
            fun()

    def add_threshold_callback(self, callback):
        self.threshold_change_callback.append(callback)

    def add_threshold_type_callback(self, callback):
        self.threshold_type_change_callback.append(callback)

    def change_min_size(self, new_min_size):
        self.minimum_size = new_min_size
        for fun in self.minimum_size_change_callback:
            fun()

    def add_min_size_callback(self, callback):
        self.minimum_size_change_callback.append(callback)

    def get_profile_list(self):
        return self.profiles.keys()

    def set_available_colormap(self, cmap_list):
        self.chosen_colormap = cmap_list
        for fun in self.callback_colormap_list:
            fun()

    def change_draw_use(self, use_draw):
        self.use_draw_result = use_draw
        for fun in self.draw_callback:
            fun()

    def add_draw_callback(self, callback):
        self.draw_callback.append(callback)

    def add_metadata_changed_callback(self, callback):
        self.metadata_changed_callback.append(callback)

    def advanced_settings_changed(self):
        for fun in self.threshold_type_change_callback:
            fun()
        for fun in self.metadata_changed_callback:
            fun()

    def metadata_changed(self):
        for fun in self.metadata_changed_callback:
            fun()


class Segment(object):
    """:type _segmented_image: np.ndarray"""
    def __init__(self, settings):
        """
        :type settings: Settings
        """
        self._settings = settings
        self._image = None
        self.draw_canvas = None
        self.draw_counter = 0
        self._gauss_image = None
        self._threshold_image = None
        self._segmented_image = None
        self._finally_segment = None
        self._sizes_array = []
        self.segmentation_change_callback = []
        self._segmentation_changed = True
        self.protect = False
        self._settings.add_threshold_callback(self.threshold_updated)
        self._settings.add_min_size_callback(self.min_size_updated)
        self._settings.add_draw_callback(self.draw_update)

    def set_image(self, image):
        self._image = image
        self._gauss_image = gaussian(self._image, 1)  # TODO radius in settings
        self._segmentation_changed = True
        self._finally_segment = np.zeros(image.shape, dtype=np.uint8)
        self.threshold_updated()

    def threshold_updated(self):
        if self.protect:
            return
        self._threshold_image = np.zeros(self._image.shape, dtype=np.uint8)
        if self._settings.use_gauss:
            image_to_threshold = self._gauss_image
        else:
            image_to_threshold = self._image
        # Define wich threshold use
        if self._settings.threshold_type == UPPER:
            def get_mask(image, threshold):
                return image <= threshold
        else:
            def get_mask(image, threshold):
                return image >= threshold

        if self._settings.threshold_layer_separate:
            print("Layer separate")
            for i in range(self._image.shape[0]):
                self._threshold_image[i][get_mask(image_to_threshold[i], self._settings.threshold_list[i])] = 1
        else:
            print("normal")
            self._threshold_image[get_mask(image_to_threshold, self._settings.threshold)] = 1
        if self._settings.mask is not None:
            self._threshold_image *= (self._settings.mask > 0)
        self.draw_update()

    def draw_update(self, canvas=None):
        if self.protect:
            return
        if canvas is not None:
            self.draw_canvas[...] = canvas[...]
            return
        if self._settings.use_draw_result:
            threshold_image = np.copy(self._threshold_image)
            threshold_image[self.draw_canvas == 1] = 1
            threshold_image[self.draw_canvas == 2] = 0
        else:
            threshold_image = self._threshold_image
        connect = sitk.ConnectedComponent(sitk.GetImageFromArray(threshold_image))
        self._segmented_image = sitk.GetArrayFromImage(sitk.RelabelComponent(connect))
        self._sizes_array = np.bincount(self._segmented_image.flat)
        self.min_size_updated()

    def min_size_updated(self):
        if self.protect:
            return
        ind = bisect(self._sizes_array[1:], self._settings.minimum_size, lambda x, y: x > y)
        # print(ind, self._sizes_array, self._settings.minimum_size)
        self._finally_segment = np.copy(self._segmented_image)
        self._finally_segment[self._finally_segment > ind] = 0
        self._segmentation_changed = True
        for fun in self.segmentation_change_callback:
            if isinstance(fun, tuple):
                fun[0](self._sizes_array[1:ind+1])
                continue
            fun()

    @property
    def segmentation_changed(self):
        """:rtype: bool"""
        return self._segmentation_changed

    def get_segmentation(self):
        self._segmentation_changed = False
        return self._finally_segment

    def get_size_array(self):
        return self._sizes_array

    def get_full_segmentation(self):
        return self._segmented_image

    def add_segmentation_callback(self, callback):
        self.segmentation_change_callback.append(callback)


def save_to_cmap(file_path, settings, segment):
    """
    :type file_path: str
    :type settings: Settings
    :type segment: Segment
    :return:
    """
    segmentation = segment.get_segmentation()
    image = np.copy(settings.image)
    if settings.threshold_type == UPPER:
        full_segmentation = segment.get_full_segmentation()
        noise_mean = np.mean(image[full_segmentation == 0])
        image = noise_mean - image
    image[segmentation == 0] = 0  # min(image[segmentation > 0].min(), 0)
    image[image < 0] = 0
    z, y, x = image.shape
    f = h5py.File(file_path, "w")
    grp = f.create_group('Chimera/image1')
    dset = grp.create_dataset("data_zyx", (z, y, x), dtype='f')
    dset[...] = image

    # Just to satisfy file format
    grp = f['Chimera']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')

    grp = f['Chimera/image1']
    grp.attrs['CLASS'] = np.string_('GROUP')
    grp.attrs['TITLE'] = np.string_('')
    grp.attrs['VERSION'] = np.string_('1.0')
    grp.attrs['step'] = np.array(settings.spacing, dtype=np.float32)

    dset.attrs['CLASS'] = np.string_('CARRY')
    dset.attrs['TITLE'] = np.string_('')
    dset.attrs['VERSION'] = np.string_('1.0')
    f.close()


def save_to_project(file_path, settings, segment):
    """
    :type file_path: str
    :type settings: Settings
    :type segment: Segment
    :return:
    """
    folder_path = tempfile.mkdtemp()
    np.save(os.path.join(folder_path, "image.npy"), settings.image)
    np.save(os.path.join(folder_path, "draw.npy"), segment.draw_canvas)
    np.save(os.path.join(folder_path, "res_mask.npy"), segment.get_segmentation())
    if settings.mask is not None:
        np.save(os.path.join(folder_path, "mask.npy"), settings.mask)
    important_data = dict()
    important_data['threshold_type'] = settings.threshold_type
    important_data['threshold_layer'] = settings.threshold_layer_separate
    important_data["threshold"] = settings.threshold
    important_data["threshold_list"] = settings.threshold_list
    important_data['use_gauss'] = settings.use_gauss
    important_data['spacing'] = settings.spacing
    important_data['minimum_size'] = settings.minimum_size
    important_data['use_draw'] = settings.use_draw_result
    with open(os.path.join(folder_path, "data.json"), 'w') as ff:
        json.dump(important_data, ff)
    """if file_path[-3:] != ".gz":
        file_path += ".gz" """
    tar = tarfile.open(file_path, 'w:bz2')
    for name in os.listdir(folder_path):
        tar.add(os.path.join(folder_path, name), name)
    tar.close()


def load_project(file_path, settings, segment):
    """
    :type file_path: str
    :type settings: Settings
    :type segment: Segment
    :return:
    """
    tar = tarfile.open(file_path, 'r:bz2')
    members = tar.getnames()
    important_data = json.load(tar.extractfile("data.json"))
    image = np.load(tar.extractfile("image.npy"))
    draw = np.load(tar.extractfile("draw.npy"))
    if "mask.npy" in members:
        mask = np.load(tar.extractfile("mask.npy"))
    else:
        mask = None
    settings.threshold = int(important_data["threshold"])

    settings.threshold_type = important_data["threshold_type"]
    settings.use_gauss = bool(important_data["use_gauss"])
    settings.spacing = \
        tuple(map(int, important_data["spacing"]))
    settings.minimum_size = int(important_data["minimum_size"])
    try:
        settings.use_draw_result = int(important_data["use_draw"])
    except KeyError:
        settings.use_draw_result = False
    segment.protect = True
    settings.add_image(image, file_path, mask)
    segment.protect = False
    if important_data["threshold_list"] is not None:
        settings.threshold_list = map(int, important_data["threshold_list"])
    else:
        settings.threshold_list = []
    settings.threshold_layer_separate = \
        bool(important_data["threshold_layer"])
    print(settings.threshold_list)
    segment.draw_update(draw)
    segment.threshold_updated()
