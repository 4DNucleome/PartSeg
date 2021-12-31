import importlib

import numpy as np
from napari.utils import Colormap

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.class_generator import SerializeClassEncoder, serialize_hook
from PartSegCore.image_operations import RadiusType


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
        if hasattr(o, "as_dict"):
            dkt = o.as_dict()
            dkt["__class__"] = o.__module__ + "." + o.__class__.__name__
            return dkt
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        return super().default(o)


class PartEncoder(ProfileEncoder):
    # pylint: disable=E0202
    def default(self, o):
        from PartSegCore.analysis.calculation_plan import CalculationPlan, CalculationTree
        from PartSegCore.analysis.measurement_calculation import MeasurementProfile

        if isinstance(o, MeasurementProfile):
            return {"__MeasurementProfile__": True, **o.as_dict()}
        if isinstance(o, CalculationPlan):
            return {"__CalculationPlan__": True, "tree": o.execution_tree, "name": o.name}
        if isinstance(o, CalculationTree):
            return {"__CalculationTree__": True, "operation": o.operation, "children": o.children}
        return super().default(o)


def part_hook(dkt):
    from PartSegCore.analysis.calculation_plan import CalculationPlan, CalculationTree
    from PartSegCore.analysis.measurement_calculation import MeasurementProfile

    if "__StatisticProfile__" in dkt:
        del dkt["__StatisticProfile__"]
        res = MeasurementProfile(**dkt)
        return res
    if "__MeasurementProfile__" in dkt:
        del dkt["__MeasurementProfile__"]
        res = MeasurementProfile(**dkt)
        return res
    if "__CalculationPlan__" in dkt:
        del dkt["__CalculationPlan__"]
        return CalculationPlan(**dkt)
    if "__CalculationTree__" in dkt:
        return CalculationTree(operation=dkt["operation"], children=dkt["children"])
    if (
        "__subtype__" in dkt
        and "statistic_profile" in dkt
        and dkt["__subtype__"]
        in (
            "PartSegCore.analysis.calculation_plan.MeasurementCalculate",
            "PartSeg.utils.analysis.calculation_plan.StatisticCalculate",
        )
    ):
        dkt["measurement_profile"] = dkt["statistic_profile"]
        del dkt["statistic_profile"]
    return profile_hook(dkt)


def profile_hook(dkt):
    """
    hook for json loading

    >>> import json
    >>> with open("some_file", 'r') as fp:
    ...     data = json.load(fp, object_hook=profile_hook)

    """
    if "__class__" in dkt:
        module_name, class_name = dkt["__class__"].rsplit(".", maxsplit=1)
        # the migration code should be called here
        try:
            del dkt["__class__"]
            module = importlib.import_module(module_name)
            return getattr(module, class_name)(**dkt)
        except Exception as e:  # skipcq: PTC-W0703`  # pylint: disable=W0703    # pragma: no cover
            dkt["__class__"] = module_name + "." + class_name
            dkt["__error__"] = e
    if "__ProfileDict__" in dkt:
        from PartSegCore.json_hooks import ProfileDict

        del dkt["__ProfileDict__"]
        res = ProfileDict(**dkt)
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
