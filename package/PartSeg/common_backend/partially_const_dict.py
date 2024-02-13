import itertools
from collections.abc import MutableMapping
from typing import Any, ClassVar, Dict, Generic, Iterator, Tuple, TypeVar, Union

from qtpy.QtCore import QObject, Signal

from PartSeg.common_backend.abstract_class import QtMeta

T = TypeVar("T")
RemovableInfo = Tuple[T, bool]


class PartiallyConstDict(QObject, MutableMapping, Generic[T], metaclass=QtMeta):
    """
    Base class for creating dict to mixin predefined and user defined variables.
    """

    item_added = Signal(object)
    """Signal with item added to dict"""
    item_removed = Signal(object)
    """Signal with item remove from dict"""
    const_item_dict: ClassVar[Dict[str, Any]] = {}
    """Dict with non removable elements"""

    def __init__(self, editable_items):
        super().__init__()
        self.editable_items = editable_items
        self._order_dict = {
            name: i for i, name in enumerate(itertools.chain(self.const_item_dict.keys(), editable_items.keys()))
        }
        self._counter = len(self._order_dict)

    def __setitem__(self, key: str, value: Union[T, RemovableInfo]) -> None:
        if key in self.const_item_dict:
            raise ValueError("Cannot write base item")
        self.editable_items[key] = value[0] if isinstance(value, tuple) else value
        self._order_dict[key] = self._counter
        self._counter += 1
        self.item_added.emit(self.editable_items[key])

    def __len__(self) -> int:
        return len(self.editable_items) + len(self.const_item_dict)

    def __iter__(self) -> Iterator[str]:
        return itertools.chain(self.const_item_dict, self.editable_items)

    def __getitem__(self, key: str) -> RemovableInfo:
        try:
            if key in self.const_item_dict:
                return self.const_item_dict[key], False
            return self.editable_items[key], True
        except KeyError as e:
            raise KeyError(f"Element {key} not found") from e

    def __delitem__(self, key: str):
        if key in self.const_item_dict:
            raise ValueError(f"Cannot delete base item {key}")
        item = self.editable_items[key]
        del self.editable_items[key]
        del self._order_dict[key]
        self.item_removed.emit(item)

    def _refresh_order(self):
        """workaround for load data problem"""
        self._order_dict = {
            name: i for i, name in enumerate(itertools.chain(self.const_item_dict.keys(), self.editable_items.keys()))
        }
        self._counter = len(self._order_dict)

    def get_position(self, key: str) -> int:
        """
        Get item position as unique int. For sorting purpose

        :raise KeyError: if element not in dict
        """
        try:
            return self._order_dict[key]
        except KeyError:
            if key not in self:
                raise
            self._order_dict[key] = self._counter
            self._counter += 1
            return self._counter - 1
