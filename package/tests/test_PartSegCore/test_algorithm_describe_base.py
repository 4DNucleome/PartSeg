from pydantic import BaseModel

from PartSegCore.algorithm_describe_base import AlgorithmProperty, _GetDescriptionClass


def test_get_description_class():
    class SampleClass:
        __test_class__ = _GetDescriptionClass()

        @classmethod
        def get_fields(self):
            return [AlgorithmProperty("test1", "Test 1", 1), AlgorithmProperty("test2", "Test 2", 2.0)]

    val = SampleClass.__test_class__
    assert val.__name__ == "__test_class__"
    assert val.__qualname__.endswith("SampleClass.__test_class__")
    assert issubclass(val, BaseModel)
    assert val.__fields__.keys() == {"test1", "test2"}
