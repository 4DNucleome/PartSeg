from partseg_utils.class_generator import BaseReadonlyClass
from typing import NamedTuple, Any
from partseg_utils.mask_create import calculate_mask, MaskProperty
import typing


class AAA(BaseReadonlyClass):
    field1: int
    field2: str
    field5: MaskProperty
    field3: Any = "ala"
    field4: float = 0.4
    field6: typing.Tuple[int] = [1,2,3]

a = AAA(1, "a", 3)
print(a.field5)

#print(calculate_mask.__annotations__)
#print(MaskProperty._source)