import logging

from qtpy.QtWidgets import QComboBox

__author__ = "Grzegorz Bokota"


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
        try:
            getattr(obj, name)
            setattr(obj, name, dic[name])
        except AttributeError as ae:
            logging.warning(ae)

def bisect(arr, val, comp):
    l = -1
    r = len(arr)
    while r - l > 1:
        e = (l + r) >> 1
        if comp(arr[e], val):
            l = e
        else:
            r = e
    return r


class SynchronizeValues(object):
    @staticmethod
    def add_synchronization(field_name, widgets):
        w = widgets[0]
        field = getattr(w, field_name)
        if isinstance(field, QComboBox):
            def synchronize(val):
                for el in widgets:
                    f = getattr(el, field_name)
                    f.setCurrentIndex(val)
            for el in widgets:
                f = getattr(el, field_name)
                f.currentIndexChanged[int].connect(synchronize)