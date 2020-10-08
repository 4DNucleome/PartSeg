import sys
import typing

import pytest
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QWidget

from PartSeg.common_backend.abstract_class import QtMeta
from PartSeg.common_backend.partially_const_dict import PartiallyConstDict


def test_object_inheritance():
    class A:
        pass

    class B(QObject, A, metaclass=QtMeta):
        pass

    assert issubclass(B, A)
    assert issubclass(B, QObject)


def test_widget_inheritance():
    class A:
        pass

    class B(QWidget, A, metaclass=QtMeta):
        pass

    assert issubclass(B, A)
    assert issubclass(B, QWidget)


@pytest.mark.skipif(sys.version_info.minor == 6, reason="Generic cannot be used with QObject in python 3.6")
def test_object_generic_inheritance():
    T = typing.TypeVar("T")

    class A:
        pass

    class B(QWidget, A, typing.Generic[T], metaclass=QtMeta):
        pass

    assert issubclass(B, A)
    assert issubclass(B, QWidget)


def test_partial_const_dict():
    class A(PartiallyConstDict[int]):
        const_item_dict = {"a": 1, "b": 2}

    assert len(A({})) == 2
