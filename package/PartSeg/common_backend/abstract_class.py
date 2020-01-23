import abc
import sys

from qtpy.QtCore import QObject


class QtMeta(type(QObject), abc.ABCMeta):
    """
    Class to solve metaclass conflict for multiple inheritance:

    ``TypeError: metaclass conflict: the metaclass of a derived class must be a
    (non-strict) subclass of the metaclass of all its bases``

    >>> class A:
    ...    pass
    ...
    >>> class Test(QObject, A, metaclass=QtMeta):
    ...    pass

    """


if sys.version_info.major == 3 and sys.version_info.minor == 6:

    def get_item(self, _item):
        return self

    QtMeta.__getitem__ = get_item
