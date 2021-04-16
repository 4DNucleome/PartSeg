import copy
import typing

import numpy as np
from napari.utils import Colormap

from PartSegCore.algorithm_describe_base import ROIExtractionProfile

from .class_generator import SerializeClassEncoder, serialize_hook
from .image_operations import RadiusType


def recursive_update_dict(main_dict: dict, other_dict: dict):
    """
    recursive update main_dict with elements of other_dict

    :param main_dict: dict to be updated recursively
    :param other_dict: source dict

    >>> dkt1 = {"test": {"test": 1, "test2": {"test4": 1}}}
    >>> dkt2 = {"test": {"test": 4, "test2": {"test2": 1}}}
    >>> recursive_update_dict(dkt1, dkt2)
    >>> dkt1
        {"test": {"test": 1, "test2": {"test4": 1, "test2": 1}}}
    """
    for key, val in other_dict.items():
        if key in main_dict and isinstance(main_dict[key], dict) and isinstance(val, dict):
            recursive_update_dict(main_dict[key], val)
        else:
            main_dict[key] = val


class ProfileDict:
    """
    Dict for storing recursive data. The path are dot separated.

    >>> dkt = ProfileDict()
    >>> dkt.set(["aa", "bb", "c1"], 7)
    >>> dkt.get(["aa", "bb", "c1"])
        7
    >>> dkt.get("aa.bb.c2", 8)
        8
    >>> dkt.get("aa.bb")
        {'c1': 7, 'c2': 8}
    """

    def __init__(self):
        self.my_dict = {}

    def update(self, ob: typing.Union["ProfileDict", dict, None] = None, **kwargs):
        """
        Update dict recursively. Use :py:func:`~.recursive_update_dict`

        :param ob: data source
        :param kwargs: data source, as keywords
        """
        if isinstance(ob, ProfileDict):
            recursive_update_dict(self.my_dict, ob.my_dict)
            recursive_update_dict(self.my_dict, kwargs)
        elif isinstance(ob, dict):
            recursive_update_dict(self.my_dict, ob)
            recursive_update_dict(self.my_dict, kwargs)
        elif ob is None:
            recursive_update_dict(self.my_dict, kwargs)

    def set(self, key_path: typing.Union[list, str], value):
        """
        Set value from dict

        :param key_path: Path to element. If is string then will be split on '.' (`key_path.split('.')`)
        :param value: Value to set.
        """
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict

        for i, key in enumerate(key_path[:-1]):
            try:
                # TODO add check if next step element is dict and create custom information
                curr_dict = curr_dict[key]
            except KeyError:
                for key2 in key_path[i:-1]:
                    curr_dict[key2] = {}
                    curr_dict = curr_dict[key2]
                    break
        curr_dict[key_path[-1]] = value

    def get(self, key_path: typing.Union[list, str], default=None):
        """
        Get value from dict.

        :param key_path: Path to element. If is string then will be split on . (`key_path.split('.')`).
        :param default: default value if element missed in dict.
        :raise KeyError: on missed element if default is not provided.
        :return: requested value
        """
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict
        for key in key_path:
            try:
                curr_dict = curr_dict[key]
            except KeyError as e:
                if default is None:
                    raise e

                val = copy.deepcopy(default)
                self.set(key_path, val)
                return val

        return curr_dict

    def verify_data(self) -> bool:
        """
        Call :py:func:`~.check_loaded_dict` on inner structures
        """
        return check_loaded_dict(self.my_dict)

    def filter_data(self):
        error_list = []
        for group, up_dkt in list(self.my_dict.items()):
            if not isinstance(up_dkt, dict):
                continue
            for key, dkt in list(up_dkt.items()):
                if not check_loaded_dict(dkt):
                    error_list.append(f"{group}.{key}")
                    del up_dkt[key]
        return error_list


class ProfileEncoder(SerializeClassEncoder):
    """
    Json encoder for :py:class:`ProfileDict`, :py:class:`RadiusType`,
     :py:class:`.SegmentationProfile` classes

    >>> import json
    >>> data = ProfileDict()
    >>> data.set("aa.bb.cc", 7)
    >>> with open("some_file", 'w') as fp:
    >>>     json.dump(data, fp, cls=ProfileEncoder)
    """

    # pylint: disable=E0202
    def default(self, o):
        """encoder implementation"""
        if isinstance(o, ProfileDict):
            return {"__ProfileDict__": True, **o.my_dict}
        if isinstance(o, RadiusType):
            return {"__RadiusType__": True, "value": o.value}
        if isinstance(o, ROIExtractionProfile):
            return {"__SegmentationProfile__": True, "name": o.name, "algorithm": o.algorithm, "values": o.values}
        if isinstance(o, Colormap):
            return {
                "__Colormap__": True,
                "name": o.name,
                "colors": o.colors.tolist(),
                "interpolation": o.interpolation,
                "controls": o.controls.tolist(),
            }
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


def profile_hook(dkt):
    """
    hook for json loading

    >>> import json
    >>> with open("some_file", 'r') as fp:
    ...     data = json.load(fp, object_hook=profile_hook)

    """
    if "__ProfileDict__" in dkt:
        del dkt["__ProfileDict__"]
        res = ProfileDict()
        res.my_dict = dkt
        return res
    if "__RadiusType__" in dkt:
        return RadiusType(dkt["value"])
    if "__SegmentationProperty__" in dkt:
        del dkt["__SegmentationProperty__"]
        res = ROIExtractionProfile(**dkt)
        return res
    if "__SegmentationProfile__" in dkt:
        del dkt["__SegmentationProfile__"]
        res = ROIExtractionProfile(**dkt)
        return res
    if (
        "__Serializable__" in dkt and dkt["__subtype__"] == "HistoryElement" and "algorithm_name" in dkt
    ):  # pragma: no cover
        # old code fix
        name = dkt["algorithm_name"]
        par = dkt["algorithm_values"]
        del dkt["algorithm_name"]
        del dkt["algorithm_values"]
        dkt["segmentation_parameters"] = {"algorithm_name": name, "values": par}
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.ColorMap":
        positions, colors = list(zip(*dkt["colormap"]))
        return Colormap(colors, controls=positions)
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.ColorPosition":
        return (dkt["color_position"], dkt["color"])
    if "__Serializable__" in dkt and dkt["__subtype__"] == "PartSegCore.color_image.base_colors.Color":
        return (dkt["red"] / 255, dkt["green"] / 255, dkt["blue"] / 255)
    if "__Colormap__" in dkt:
        del dkt["__Colormap__"]
        if dkt["controls"][0] != 0:
            dkt["controls"].insert(0, 0)
            dkt["colors"].insert(0, dkt["colors"][0])
        if dkt["controls"][-1] != 1:
            dkt["controls"].append(1)
            dkt["colors"].append(dkt["colors"][-1])
        return Colormap(**dkt)

    return serialize_hook(dkt)


def check_loaded_dict(dkt) -> bool:
    """
    Recursive check if dict `dkt` or any sub dict contains '__error__' key.

    :param dkt: dict to check
    """
    if not isinstance(dkt, dict):
        return True
    if "__error__" in dkt:
        return False
    return all(check_loaded_dict(val) for val in dkt.values())
