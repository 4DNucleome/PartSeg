import copy
import sys
import typing

from .class_generator import SerializeClassEncoder, serialize_hook
from .image_operations import RadiusType


class ProfileDict(object):
    def __init__(self):
        self.my_dict = dict()

    def update(self, ob=None, *args, **kwargs):
        if isinstance(ob, ProfileDict):
            self.my_dict.update(ob.my_dict)
            self.my_dict.update(*args, **kwargs)
        elif ob is None:
            self.my_dict.update(**kwargs)
        else:
            self.my_dict.update(ob, **kwargs)

    def set(self, key_path: typing.Union[list, str], value):
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict

        for i, key in enumerate(key_path[:-1]):
            try:
                curr_dict = curr_dict[key]
            except KeyError:
                for key2 in key_path[i:-1]:
                    curr_dict[key2] = dict()
                    curr_dict = curr_dict[key2]
                    break
        curr_dict[key_path[-1]] = value

    def get(self, key_path: typing.Union[list, str], default):
        if isinstance(key_path, str):
            key_path = key_path.split(".")
        curr_dict = self.my_dict
        for i, key in enumerate(key_path):
            try:
                curr_dict = curr_dict[key]
            except KeyError as e:
                if default is not None:
                    val = copy.deepcopy(default)
                    self.set(key_path, val)
                    return val
                else:
                    print(f"{key_path}: {curr_dict.items()}", file=sys.stderr)
                    raise e
        return curr_dict

    def verify_data(self):
        return check_loaded_dict(self.my_dict)

    def filter_data(self):
        error_list = []
        for group, up_dkt in list(self.my_dict.items()):
            for key, dkt in list(up_dkt.items()):
                if not check_loaded_dict(dkt):
                    error_list.append(f"{group}.{key}")
                    del up_dkt[key]
        return error_list


class ProfileEncoder(SerializeClassEncoder):
    def default(self, o):
        if isinstance(o, ProfileDict):
            return {"__ProfileDict__": True, **o.my_dict}
        if isinstance(o, RadiusType):
            return {"__RadiusType__": True, "value": o.value}
        return super().default(o)


def profile_hook(dkt):
    if "__ProfileDict__" in dkt:
        del dkt["__ProfileDict__"]
        res = ProfileDict()
        res.my_dict = dkt
        return res
    if "__RadiusType__" in dkt:
        return RadiusType(dkt["value"])
    return serialize_hook(dkt)


def check_loaded_dict(dkt):
    if not isinstance(dkt, dict):
        return True
    if "__error__" in dkt:
        return False
    for val in dkt.values():
        if not check_loaded_dict(val):
            return False
    return True
