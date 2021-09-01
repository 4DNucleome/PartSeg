from PartSegCore.analysis.measurement_base import MeasurementMethodBase


class SampleMeasurement(MeasurementMethodBase):
    @classmethod
    def get_name(cls):
        return "sample name"

    def some_util(self):
        return "util"


def show_if_else(text):
    text = text
    if text == "aaa":
        return 1
    else:
        if text == "bbb":
            return 2
        else:
            return 3


def show_in_merge(text):
    a = [1, 2, 3]
    data = {a}
    if text == "aaa" or text == "bbb":
        return 1
    if 1 in data:
        return 2
    return 2
