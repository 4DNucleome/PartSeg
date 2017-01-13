from collections import namedtuple, OrderedDict
from utils import class_to_dict
import logging
import SimpleITK as sitk
import numpy as np

import auto_fit as af

SettingsValue = namedtuple("SettingsValue", ["function_name", "help_message", "arguments"])
Leaf = namedtuple("Leaf", ["name", "dict"])
Node = namedtuple("Node", ["left", 'op', 'right'])


class StatisticProfile(object):

    STATISTIC_DICT = {
        "Volume": SettingsValue("calculate_volume", "Calculate volume of current segmentation", None),
        "Volume per component": SettingsValue("calculate_component_volume", "Calculate volume of each component "
                                              "of cohesion of current segmentation", None),
        "Mass": SettingsValue("calculate_mass", "Sum of pixel brightness for current segmentation", None),
        "Mass per component": SettingsValue("calculate_component_mass", "Sum of pixel brightness of each component of"
                                            " cohesion for current segmentation", None),
        "Border surface": SettingsValue("calculate_border_surface",
                                        "Calculating surface of current segmentation", None),
        "Maximum pixel brightness": SettingsValue(
            "maximum_brightness", "Calculate maximum brightness of pixel for current segmentation", None),
        "Minimum pixel brightness": SettingsValue(
            "minimum_brightness", "Calculate minimum brightness of pixel for current segmentation", None),
        "Median pixel brightness": SettingsValue(
            "median_brightness", "Calculate median brightness of pixel for current segmentation", None),
        "Mean pixel brightness": SettingsValue(
            "mean_brightness", "Calculate median brightness of pixel for current segmentation", None),
        "Standard deviation of pixel brightness": SettingsValue(
            "std_brightness", "Calculate  standard deviation of pixel brightness for current segmentation", None),
        "Standard deviation of Noise": SettingsValue(
            "std_noise", "Calculate standard deviation of pixel brightness outside current segmentation", None),
        "Moment of inertia": SettingsValue("moment_of_inertia", "Calculate moment of inertia for segmented structure."
                                           "Has one parameter thr (threshold). Only values above it are used "
                                           "in calculation", None),
        "Border Mass": SettingsValue("border_mass", "Calculate mass for elements in radius (in physical units)"
                                                    " from mask", {"radius": int}),
        "Border Volume": SettingsValue("border_volume", "Calculate volumes for elements in radius (in physical units)"
                                                        " from mask", {"radius": int})
    }
    PARAMETERS = ["name", "chosen_fields", "reversed_brightness", "use_gauss_image", "name_prefix"]

    def __init__(self, name, chosen_fields, reversed_brightness, settings=None, use_gauss_image=False, name_prefix=""):
        self.name = name
        self.chosen_fields = []
        for cf_val in chosen_fields:
            user_name = cf_val[1]
            if isinstance(cf_val[0], str):
                tree = self.parse_statistic(cf_val[0])
            else:
                tree = self.rebuild_tree(cf_val[0])
            self.chosen_fields.append((tree, user_name, None))
        self.settings = settings
        self.reversed_brightness = reversed_brightness
        self.use_gauss_image = use_gauss_image
        self.name_prefix = name_prefix

    def __str__(self):
        text = "Profile name: {}\n".format(self.name)
        text += "Reversed image [{}]\n".format(self.reversed_brightness)
        text += "Gaussed image [{}]\n".format(self.use_gauss_image)
        if self.name_prefix != "":
            text += "Name prefix: {}\n".format(self.name_prefix)
        text += "statistics list:\n"
        for el in self.chosen_fields:
            if el[2] is not None:
                text += "{}: {}\n".format(el[1], el[2])
            else:
                text += "{}\n".format(el[1])
        return text

    def get_parameters(self):
        return class_to_dict(self, *self.PARAMETERS)

    def rebuild_tree(self, l):
        if len(l) == 2:
            return Leaf(*l)
        else:
            return Node(self.rebuild_tree(l[0]), l[1], self.rebuild_tree(l[2]))

    def flat_tree(self, t):
        if isinstance(t, Leaf):
            res = ""
            if t.dict is not None and len(t.dict) > 0:
                for name, val in t.dict.items():
                    res += "{}={},".format(name, val)
                return "{}[{}]".format(t.name, res[:-1])
            return t.name
        elif isinstance(t, Node):
            if isinstance(t.left, Node):
                beg = "({})"
            else:
                beg = "{}"
            if isinstance(t.right, Node):
                end = "({})"
            else:
                end = "{}"
            return (beg+"{}"+end).format(self.flat_tree(t.left), t.op, self.flat_tree(t.right))

    @staticmethod
    def tokenize(text):
        special = ["(", ")", "[", "]", "/", "+", ","]
        res = []
        temp_str = ""
        for l in text:
            if l in special:
                if temp_str != "":
                    res.append(temp_str)
                    temp_str = ""
                res.append(l)
            else:
                temp_str += l
        if temp_str != "":
            res.append(temp_str)
        return res

    def build_tree(self, tokens):
        res = []
        final_res = res
        pos = 0
        while True:
            if pos == len(tokens):
                break
            if tokens[pos] == ")":
                pos += 1
                break
            if tokens[pos] == "/":
                final_res = [res[:], "/"]
                res = []
                final_res.append(res)
                pos += 1
            if tokens[pos] in "[],":
                pos += 1
                continue
            if tokens[pos] == "(":
                sub_tree, pos_shift = self.build_tree(tokens[pos+1:])
                pos += pos_shift+1
                res.extend(sub_tree)
                continue
            res.append(tokens[pos])
            pos += 1
        return final_res, pos

    def tree_to_dict_tree(self, tree):
        if isinstance(tree[0], list):
            left_tree = self.tree_to_dict_tree(tree[0])
            right_tree = self.tree_to_dict_tree(tree[2])
            return Node(left_tree, tree[1], right_tree)
        else:
            name = tree[0]
            base_stat = self.STATISTIC_DICT[name]
            d = dict()
            for el in tree[1:]:
                sp = el.split("=")
                d[sp[0]] = base_stat.arguments[sp[0]](sp[1])
            return Leaf(name, d)

    def parse_statistic(self, text):
        tokens = self.tokenize(text)

        tree, l = self.build_tree(tokens)
        return self.tree_to_dict_tree(tree)

    def calculate_tree(self, node, help_dict, kwargs):
        """
        :type node: Leaf | Node
        :type help_dict: dict
        :type kwargs: dict
        :return: float
        """
        if isinstance(node, Leaf):
            fun_name = self.STATISTIC_DICT[node.name][0]
            kw = dict(kwargs)
            kw.update(node.dict)
            hash_str = "{}: {}".format(fun_name, kw)
            if hash_str in help_dict:
                return help_dict[hash_str]
            fun = getattr(self, fun_name)
            val = fun(**kw)
            help_dict[hash_str] = val
            return val
        elif isinstance(node, Node):
            left_res = self.calculate_tree(node.left, help_dict, kwargs)
            right_res = self.calculate_tree(node.right, help_dict, kwargs)
            if node.op == "/":
                return left_res/right_res
        logging.error("Wrong statistics: {}".format(node))
        return 1

    def calculate(self, image, gauss_image, mask, full_mask, base_mask):
        result = OrderedDict()
        if self.use_gauss_image:
            image = gauss_image.astype(np.float)
        else:
            image = image.astype(np.float)
        if self.reversed_brightness:
            noise_mean = np.mean(image[full_mask == 0])
            image = noise_mean - image
        help_dict = dict()
        kw = {"image": image, "mask": mask, "base_mask": base_mask, "full_mask": full_mask}
        for tree, user_name, params in self.chosen_fields:
            try:
                result[user_name] = self.calculate_tree(tree, help_dict, kw)
            except ZeroDivisionError:
                result[user_name] = "Div by zero"
            except TypeError:
                result[user_name] = "None div"
        return result

    @staticmethod
    def pixel_volume(x):
        return x[0] * x[1] * x[2]

    def calculate_volume(self, mask, **_):
        print("Volume {}".format(np.max(mask)))
        return np.count_nonzero(mask) * self.pixel_volume(self.settings.voxel_size)

    def calculate_component_volume(self, mask, **_):
        return np.bincount(mask.flat)[1:] * self.pixel_volume(self.settings.voxel_size)

    @staticmethod
    def calculate_mass(mask, image, **_):
        if np.any(mask):
            return np.sum(image[mask > 0])
        return 0

    @staticmethod
    def calculate_component_mass(mask, image, **_):
        res = []
        for i in range(1, mask.max()+1):
            res.append(np.sum(image[mask == i]))
        return res

    def calculate_border_surface(self, mask, **_):
        return calculate_volume_surface(mask, self.settings.voxel_size)

    @staticmethod
    def maximum_brightness(mask, image, **_):
        if np.any(mask):
            return np.max(image[mask > 0])
        else:
            return None

    @staticmethod
    def minimum_brightness(mask, image, **_):
        if np.any(mask):
            return np.min(image[mask > 0])
        else:
            return None

    @staticmethod
    def median_brightness(mask, image, **_):
        if np.any(mask):
            return np.median(image[mask > 0])
        else:
            return None

    @staticmethod
    def std_brightness(mask, image, **_):
        if np.any(mask):
            return np.std(image[mask > 0])
        else:
            return None

    @staticmethod
    def mean_brightness(mask, image, **_):
        if np.any(mask):
            return np.mean(image[mask > 0])
        else:
            return None

    @staticmethod
    def std_noise(mask, base_mask, image, **_):
        if np.any(mask):
            if base_mask is not None:
                return np.std(image[(mask == 0) * (base_mask > 0)])
            else:
                return np.std(image[mask == 0])
        else:
            return None

    def moment_of_inertia(self, image, mask, **_):
        if image.ndim != 3:
            return None
        img = np.copy(image)
        img[mask == 0] = 0
        return af.calculate_density_momentum(img, self.settings.voxel_size,)

    def border_mask(self, base_mask, radius, **_):
        if base_mask is None:
            return None
        base_mask = np.array(base_mask > 0)
        base_mask = base_mask.astype(np.uint8)
        border = sitk.LabelContour(sitk.GetImageFromArray(base_mask))
        border.SetSpacing(self.settings.voxel_size)
        dilated_border = sitk.GetArrayFromImage(sitk.BinaryDilate(border, radius))
        dilated_border[base_mask == 0] = 0
        return dilated_border

    def border_mass(self, image, mask, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (mask > 0))
        if np.any(final_mask):
            return np.sum(image[final_mask])
        return 0

    def border_volume(self, mask, **kwargs):
        border_mask = self.border_mask(**kwargs)
        if border_mask is None:
            return None
        final_mask = np.array((border_mask > 0) * (mask > 0))
        return np.count_nonzero(final_mask) * self.pixel_volume(self.settings.voxel_size)


def calculate_volume_surface(volume_mask, voxel_size):
    border_surface = 0
    surf_im = np.array(volume_mask).astype(np.uint8)
    border_surface += np.count_nonzero(np.logical_xor(surf_im[1:], surf_im[:-1])) * voxel_size[1] * voxel_size[2]
    border_surface += np.count_nonzero(np.logical_xor(surf_im[:, 1:], surf_im[:, :-1])) * voxel_size[0] * voxel_size[2]
    if len(surf_im.shape) == 3:
        border_surface += np.count_nonzero(np.logical_xor(surf_im[:, :, 1:], surf_im[:, :, :-1])) * voxel_size[0] * \
                          voxel_size[1]
    return border_surface
