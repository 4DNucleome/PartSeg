import logging
import os
import sys
import textwrap
import typing
import uuid
from abc import abstractmethod
from copy import copy, deepcopy
from enum import Enum

from ..algorithm_describe_base import ROIExtractionProfile
from ..class_generator import BaseSerializableClass, enum_register
from ..mask_create import MaskProperty
from ..universal_const import Units
from . import analysis_algorithm_dict
from .measurement_calculation import MeasurementProfile


class MaskBase:
    """
    Base class for mask in calculation plan.

    :ivar str ~.name: name of mask
    """

    name: str


# MaskCreate = namedtuple("MaskCreate", ['name', 'radius'])


class RootType(Enum):
    """Defines root type which changes of data available on begin of calculation"""

    Image = 0  #: raw image
    Project = 1  #: PartSeg project with defined segmentation
    Mask_project = 2  #: Project from mask segmentation. It contains multiple elements.

    def __str__(self):
        return self.name.replace("_", " ")


enum_register.register_class(RootType)


class MaskCreate(MaskBase, BaseSerializableClass):
    """
    Description of mask creation in calculation plan.

    :ivar str ~.name: name of mask
    :ivar str ~.mask_property: instance of :py:class:`.MaskProperty`
    """

    mask_property: MaskProperty

    def __str__(self):
        return f"Mask create: {self.name}\n" + str(self.mask_property).split("\n", 1)[1]


class MaskUse(MaskBase, BaseSerializableClass):
    """
    Reuse of already defined mask
    Will be deprecated in short time
    """


class MaskSum(MaskBase, BaseSerializableClass):
    """
    Description of OR operation on mask

    :ivar str ~.name: name of mask
    :ivar str ~.mask1: first mask name
    :ivar str ~.mask2: second mask name
    """

    mask1: str
    mask2: str


class MaskIntersection(MaskBase, BaseSerializableClass):
    """
    Description of AND operation on mask

    :ivar str ~.name: name of mask
    :ivar str ~.mask1: first mask name
    :ivar str ~.mask2: second mask name
    """

    mask1: str
    mask2: str


class Save(BaseSerializableClass):
    """
    Save operation description

    :ivar str ~.suffix: suffix for saved file
    :ivar str ~.directory: name of subdirectory to save
    :ivar str ~.algorithm: name of save method
    :ivar str ~.short_name: short name of save method
    :ivar dict ~.values: parameters specific for save method
    """

    suffix: str
    directory: str
    algorithm: str
    short_name: str
    values: dict


class MeasurementCalculate(BaseSerializableClass):
    """
    Measurement calculation description

    :ivar int ~.channel: on which channel measurements should be calculated
    :ivar Units ~.units: Type of units in which results of measurements should be represented
    :ivar MeasurementProfile ~.statistic_profile: description of measurements
    :ivar str name_prefix: prefix of column names
    """

    __old_names__ = "StatisticCalculate"
    channel: int
    units: Units
    measurement_profile: MeasurementProfile
    name_prefix: str
    # TODO rename statistic_profile to measurement_profile

    # noinspection PyOverloads,PyMissingConstructor
    # pylint: disable=W0104
    # pragma: no cover
    @typing.overload
    def __init__(self, channel: int, units: Units, measurement_profile: MeasurementProfile, name_prefix: str):
        ...

    @property
    def name(self):
        """name of used MeasurementProfile"""
        return self.measurement_profile.name

    def __str__(self):
        channel = "Like segmentation" if self.channel == -1 else str(self.channel)
        desc = str(self.measurement_profile).split("\n", 1)[1]
        return f"MeasurementCalculate \nChannel: {channel}\nUnits: {self.units}\n{desc}\n"


def get_save_path(op: Save, calculation: "FileCalculation") -> str:
    """
    Calculate save path base on proceeded file path and save operation parameters.
    It assume that save algorithm is registered in :py:data:`PartSegCore.analysis.save_functions.save_dict`

    :param op: operation to do
    :param calculation: information about calculation
    :return: save path
    """
    from PartSegCore.analysis.save_functions import save_dict

    extension = save_dict[op.algorithm].get_default_extension()
    rel_path = os.path.relpath(calculation.file_path, calculation.base_prefix)
    rel_path = os.path.splitext(rel_path)[0]
    if op.directory:
        return os.path.join(calculation.result_prefix, rel_path, op.suffix + extension)
    return os.path.join(calculation.result_prefix, rel_path + op.suffix + extension)


class MaskMapper:
    """
    Base class for obtaining mask from computer disc

    :ivar ~.name: mask name
    """

    name: str

    @abstractmethod
    def get_mask_path(self, file_path: str) -> str:
        """
        Calculate mask path based od file_path

        :param file_path: path to proceeded file
        """

    @abstractmethod
    def get_parameters(self):
        """Parameters for serialize"""

    @staticmethod
    def is_ready() -> bool:
        """Check if this mask mapper can be used"""
        return True


class MaskSuffix(MaskMapper, BaseSerializableClass):
    """
    Description of mask form file obtained by adding suffix to image file path

    :ivar str ~.name: mask name
    :ivar str ~.suffix: mask file path suffix
    """

    suffix: str

    # noinspection PyMissingConstructor,PyOverloads
    # pylint: disable=W0104
    @typing.overload
    def __init__(self, name: str, suffix: str):  # pragma: no cover
        ...

    def get_mask_path(self, file_path: str) -> str:
        base, ext = os.path.splitext(file_path)
        return base + self.suffix + ext

    def get_parameters(self):
        return {"name": self.name, "suffix": self.suffix}


class MaskSub(MaskMapper, BaseSerializableClass):
    """
    Description of mask form file obtained by substitution

    :ivar str ~.name: mask name
    :ivar str ~.base: string to be searched
    :ivar str ~.repr: string to be put instead of ``base``
    """

    base: str
    rep: str

    # noinspection PyMissingConstructor,PyOverloads
    # pylint: disable=W0104
    @typing.overload
    def __init__(self, name: str, base: str, rep: str):  # pragma: no cover
        ...

    def get_mask_path(self, file_path: str) -> str:
        dir_name, filename = os.path.split(file_path)
        filename = filename.replace(self.base, self.rep)
        return os.path.join(dir_name, filename)

    def get_parameters(self):
        return {"name": self.name, "base": self.base, "rep": self.rep}


class MaskFile(MaskMapper, BaseSerializableClass):
    # TODO Check implementation
    path_to_file: str
    name_dict: typing.Optional[dict] = None

    # noinspection PyMissingConstructor,PyOverloads
    # pylint: disable=W0104
    @typing.overload
    def __init__(self, name: str, path_to_file: str, name_dict: typing.Optional[dict] = None):  # pragma: no cover
        ...

    def is_ready(self) -> bool:
        return os.path.exists(self.path_to_file)

    def get_mask_path(self, file_path: str) -> str:
        if self.name_dict is None:
            self.parse_map()
        try:
            return self.name_dict[os.path.normpath(file_path)]
        except KeyError:
            return ""
        except AttributeError:
            return ""

    def get_parameters(self):
        return {"name": self.name, "path_to_file": self.path_to_file}

    def set_map_path(self, value):
        self.path_to_file = value

    def parse_map(self, sep=";"):
        if not os.path.exists(self.path_to_file):
            logging.error(f"File does not exists: {self.path_to_file}")
            raise ValueError(f"File for mapping mask does not exists: {self.path_to_file}")
        with open(self.path_to_file) as map_file:
            dir_name = os.path.dirname(self.path_to_file)
            for i, line in enumerate(map_file):
                try:
                    file_name, mask_name = line.split(sep)
                except ValueError:
                    logging.error(f"Error in parsing map file\nline {i}\n{line}\nfrom file{self.path_to_file}")
                    continue
                file_name = file_name.strip()
                mask_name = mask_name.strip()
                if not os.path.abspath(file_name):
                    file_name = os.path.normpath(os.path.join(dir_name, file_name))
                if not os.path.abspath(mask_name):
                    mask_name = os.path.normpath(os.path.join(dir_name, mask_name))
                self.name_dict[file_name] = mask_name


class Operations(Enum):
    """Global operations"""

    reset_to_base = 1
    # leave_the_biggest = 2


class PlanChanges(Enum):
    """History elements"""

    add_node = 1  #:
    remove_node = 2  #:
    replace_node = 3  #:


class CalculationTree:
    """
    Structure for describe calculation structure
    """

    def __init__(
        self,
        operation: typing.Union[BaseSerializableClass, ROIExtractionProfile, MeasurementCalculate, RootType],
        children: typing.List["CalculationTree"],
    ):
        if operation == "root":
            operation = RootType.Image
        self.operation = operation
        self.children = children

    def __str__(self):
        return f"{self.operation}:\n[{'n'.join([str(x) for x in self.children])}]"

    def __repr__(self):
        return f"CalculationTree(operation={repr(self.operation)}, children={self.children})"


class NodeType(Enum):
    """Type of node in calculation"""

    segment = 1  #: segmentation
    mask = 2  #: mask creation
    measurement = 3  #: measurement calculation
    root = 4  #: root of calculation
    save = 5  #: save operation
    none = 6  #: other, like description
    file_mask = 7  #: mask load


class BaseCalculation:
    """
    Base description of calculation needed for single file

    :ivar str ~.base_prefix: path prefix which should be used to calculate relative path of processed files
    :ivar str ~.result_prefix: path prefix for saving structure
    :ivar str ~.measurement_file_path: path to file in which result of measurement should be saved
    :ivar str ~.sheet_name: name of sheet in excel file
    :ivar CalculationPlan ~.calculation_plan: plan of calculation
    :ivar str uuid: ~.uuid of whole calculation
    :ivar ~.voxel_size: default voxel size (for files which do not contains this information in metadata
    """

    def __init__(
        self,
        base_prefix: str,
        result_prefix: str,
        measurement_file_path: str,
        sheet_name: str,
        calculation_plan: "CalculationPlan",
        voxel_size: typing.Sequence[float],
    ):
        self.base_prefix = base_prefix
        self.result_prefix = result_prefix
        self.measurement_file_path = measurement_file_path
        self.sheet_name = sheet_name
        self.calculation_plan = calculation_plan
        self.uuid = uuid.uuid4()
        self.voxel_size = voxel_size

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(calculation_plan={self.calculation_plan}, voxel_size={self.voxel_size}, "
            f"base_prefix={self.base_prefix}, result_prefix={self.base_prefix}, "
            f"measurement_file_path{self.measurement_file_path}, sheet_name={self.sheet_name})"
        )


class Calculation(BaseCalculation):
    """
    Description of whole calculation. Extended with list of all files to proceed

    :ivar str ~.base_prefix: path prefix which should be used to calculate relative path of processed files
    :ivar str ~.result_prefix: path prefix for saving structure
    :ivar str ~.measurement_file_path: path to file in which result of measurement should be saved
    :ivar str ~.sheet_name: name of sheet in excel file
    :ivar CalculationPlan ~.calculation_plan: plan of calculation
    :ivar str uuid: ~.uuid of whole calculation
    :ivar ~.voxel_size: default voxel size (for files which do not contains this information in metadata
    :ivar typing.List[str] ~.file_list: list of files to be proceed
    """

    def __init__(
        self, file_list, base_prefix, result_prefix, measurement_file_path, sheet_name, calculation_plan, voxel_size
    ):
        super().__init__(base_prefix, result_prefix, measurement_file_path, sheet_name, calculation_plan, voxel_size)
        self.file_list: typing.List[str] = file_list

    def get_base_calculation(self) -> BaseCalculation:
        """Extract py:class:`BaseCalculation` from instance."""
        base = BaseCalculation(
            self.base_prefix,
            self.result_prefix,
            self.measurement_file_path,
            self.sheet_name,
            self.calculation_plan,
            self.voxel_size,
        )
        base.uuid = self.uuid
        return base


class FileCalculation:
    """
    Description of single file calculation
    """

    def __init__(self, file_path: str, calculation: BaseCalculation):
        self.file_path = file_path
        self.calculation = calculation

    @property
    def base_prefix(self):
        """path prefix which should be used to calculate relative path of processed files"""
        return self.calculation.base_prefix

    @property
    def result_prefix(self):
        """path prefix for saving structure"""
        return self.calculation.result_prefix

    @property
    def calculation_plan(self):
        """plan of calculation"""
        return self.calculation.calculation_plan

    @property
    def uuid(self):
        """uuid of whole calculation"""
        return self.calculation.uuid

    @property
    def voxel_size(self):
        """default voxel size (for files which do not contains this information in metadata"""
        return self.calculation.voxel_size

    def __repr__(self):
        return f"FileCalculation(file_path={self.file_path}, calculation={self.calculation})"


class CalculationPlan:
    """
    Clean description Calculation plan.

    :type current_pos: list[int]
    :type name: str
    :type segmentation_count: int
    :type execution_tree: CalculationTree
    """

    correct_name = {
        MaskCreate.__name__: MaskCreate,
        MaskUse.__name__: MaskUse,
        Save.__name__: Save,
        MeasurementCalculate.__name__: MeasurementCalculate,
        ROIExtractionProfile.__name__: ROIExtractionProfile,
        MaskSuffix.__name__: MaskSuffix,
        MaskSub.__name__: MaskSub,
        MaskFile.__name__: MaskFile,
        Operations.__name__: Operations,
        MaskIntersection.__name__: MaskIntersection,
        MaskSum.__name__: MaskSum,
        RootType.__name__: RootType,
    }

    def __init__(self, tree: typing.Optional[CalculationTree] = None, name: str = ""):
        if tree is None:
            self.execution_tree = CalculationTree(RootType.Image, [])
        else:
            self.execution_tree = tree
        self.segmentation_count = 0
        self.name = name
        self.current_pos = []
        self.changes = []
        self.current_node = None

    def get_root_type(self):
        return self.execution_tree.operation

    def set_root_type(self, root_type: RootType):
        self.execution_tree.operation = root_type

    def __str__(self):
        return f"CalculationPlan<{self.name}>\n{self.execution_tree}"

    def __repr__(self):
        return f"CalculationPlan(name={repr(self.name)}, execution_tree={repr(self.execution_tree)})"

    def get_measurements(self, node: typing.Optional[CalculationTree] = None) -> typing.List[MeasurementCalculate]:
        """
        Get all measurement Calculation bellow given node

        :param node: Node for start, if absent then start from plan root
        :return: list of measurements
        """
        if node is None:
            node = self.execution_tree
        if isinstance(node.operation, MeasurementCalculate):
            return [node.operation]

        res = []
        for el in node.children:
            res.extend(self.get_measurements(el))
        return res

    def get_changes(self):
        ret = self.changes
        self.changes = []
        return ret

    def position(self):
        return self.current_pos

    def set_position(self, value):
        self.current_pos = value
        self.current_node = None

    def clean(self):
        self.execution_tree = CalculationTree(RootType.Image, [])
        self.current_pos = []

    def __copy__(self):
        return CalculationPlan(name=self.name, tree=deepcopy(self.execution_tree))

    def __deepcopy__(self, memo):
        return CalculationPlan(name=self.name, tree=deepcopy(self.execution_tree))

    def get_node(self, search_pos=None):
        """
        :param search_pos:
        :return: CalculationTree
        """
        node = self.execution_tree
        if search_pos is None:
            if self.current_node is not None:
                return self.current_node
            search_pos = self.current_pos
        for pos in search_pos:
            node = node.children[pos]
        return node

    def get_mask_names(self, node=None):
        """
        :type node: CalculationTree
        :param node:
        :return: set[str]
        """
        if node is None:
            node = self.get_node()
        res = set()
        if isinstance(node.operation, (MaskCreate, MaskMapper)):
            res.add(node.operation.name)
        for el in node.children:
            res |= self.get_mask_names(el)
        return res

    def get_file_mask_names(self):
        node = self.get_node()
        used_mask = self.get_reused_mask()
        tree_mask_names = self.get_mask_names(node)
        return used_mask & tree_mask_names, tree_mask_names

    @staticmethod
    def _get_reused_mask(node):
        """
        :type node: CalculationTree
        :param node:
        :return:
        """
        used_mask = set()
        for el in node.children:
            if isinstance(el.operation, MaskUse):
                used_mask.add(el.operation.name)
            elif isinstance(el.operation, (MaskSum, MaskIntersection)):
                used_mask.add(el.operation.mask1)
                used_mask.add(el.operation.mask2)
        return used_mask

    def get_reused_mask(self) -> set:
        return self._get_reused_mask(self.execution_tree)

    def get_node_type(self) -> NodeType:
        if self.current_pos is None:
            return NodeType.none
        if not self.current_pos:
            return NodeType.root
        # print("Pos {}".format(self.current_pos))
        node = self.get_node()
        if isinstance(node.operation, (MaskMapper, MaskIntersection, MaskSum)):
            return NodeType.file_mask
        if isinstance(node.operation, MaskCreate):
            return NodeType.mask
        if isinstance(node.operation, MeasurementCalculate):
            return NodeType.measurement
        if isinstance(node.operation, ROIExtractionProfile):
            return NodeType.segment
        if isinstance(node.operation, Save):
            return NodeType.save
        if isinstance(node.operation, MaskUse):
            return NodeType.file_mask
        if isinstance(node.operation, Operations) and node.operation == Operations.reset_to_base:
            return NodeType.mask
        raise ValueError(f"[get_node_type] unknown node type {node.operation}")

    def add_step(self, step):
        if self.current_pos is None:
            return
        node = self.get_node()
        node.children.append(CalculationTree(step, []))
        if isinstance(step, ROIExtractionProfile):
            self.segmentation_count += 1
        self.changes.append((self.current_pos, node.children[-1], PlanChanges.add_node))

    def replace_step(self, step):
        if self.current_pos is None:
            return
        node = self.get_node()
        node.operation = step
        self.changes.append((self.current_pos, node, PlanChanges.replace_node))

    def replace_name(self, name):
        if self.current_pos is None:
            return
        node = self.get_node()
        node.operation = node.operation.replace_(name=name)
        self.changes.append((self.current_pos, node, PlanChanges.replace_node))

    def has_children(self):
        node = self.get_node()
        return len(node.children) > 0

    def remove_step(self):
        path = copy(self.current_pos)
        if not path:
            return
        pos = path[-1]
        parent_node = self.get_node(path[:-1])
        del parent_node.children[pos]
        self.changes.append((self.current_pos, None, PlanChanges.remove_node))
        self.current_pos = self.current_pos[:-1]

    def is_segmentation(self):
        return self.segmentation_count > 0

    def set_name(self, text):
        self.name = text

    def get_execution_tree(self):
        return self.execution_tree

    def _get_save_list(self, node):
        """
        :type node: CalculationTree
        :param node:
        :return:
        """
        if isinstance(node.operation, Save):
            return [node.operation]

        res = []
        for chl in node.children:
            res.extend(self._get_save_list(chl))
        return res

    def get_save_list(self):
        return self._get_save_list(self.execution_tree)

    def get_list_file_mask(self):
        """
        :return: list[MaskMapper]
        """
        return [el.operation for el in self.execution_tree.children if isinstance(el.operation, MaskMapper)]

    def set_path_to_mapping_file(self, num, path):
        for el in self.execution_tree.children:
            if num == 0:
                el.operation.path_to_file = path
                return el.operation
            if isinstance(el.operation, MaskFile):
                num -= 1

    @classmethod
    def dict_load(cls, data_dict):
        res_plan = cls()
        name = data_dict["name"]
        res_plan.set_name(name)
        execution_tree = data_dict["execution_tree"]
        for pos, el, _ in execution_tree:
            res_plan.current_pos = pos[:-1]
            try:
                res_plan.add_step(CalculationPlan.correct_name[el["type"]](**el["values"]))
            except TypeError as e:
                logging.warning(el["type"])
                raise e
        res_plan.changes = []
        return res_plan

    @staticmethod
    def get_el_name(el):
        """
        :param el: Plan element
        :return: str
        """
        if el.__class__.__name__ not in CalculationPlan.correct_name.keys():
            print(el, el.__class__.__name__, file=sys.stderr)
            raise ValueError(f"Unknown type {el.__class__.__name__}")
        if isinstance(el, RootType):
            return f"Root: {el}"
        if isinstance(el, Operations) and el == Operations.reset_to_base:
            return "reset project to base image with mask"
        if isinstance(el, ROIExtractionProfile):
            return f"Segmentation: {el.name}"
        if isinstance(el, MeasurementCalculate):
            if el.name_prefix == "":
                return f"Measurement: {el.name}"
            return f"Measurement: {el.name} with prefix: {el.name_prefix}"
        if isinstance(el, MaskCreate):
            if el.name != "":
                return f"Create mask: {el.name}"
            return "Create mask:"
        if isinstance(el, MaskUse):
            return f"Use mask: {el.name}"
        if isinstance(el, MaskSuffix):
            return f"File mask: {el.name} with suffix {el.suffix}"
        if isinstance(el, MaskSub):
            return f"File mask: {el.name} substitution {el.base} on {el.rep}"
        if isinstance(el, MaskFile):
            return f"File mapping mask: {el.name}, {el.path_to_file}"
        if isinstance(el, Save):
            base = el.short_name
            if el.directory:
                text = f"Save {base} in directory with name " + el.suffix
            else:
                if el.suffix != "":
                    text = "Save " + base + " with suffix " + el.suffix
                else:
                    text = "Save " + base
            return text
        if isinstance(el, MaskIntersection):
            if el.name == "":
                return f"Mask intersection of mask {el.mask1} and {el.mask2}"
            return f"Mask {el.name} intersection of mask {el.mask1} and {el.mask2}"
        if isinstance(el, MaskSum):
            if el.name == "":
                return f"Mask sum of mask {el.mask1} and {el.mask2}"
            return f"Mask {el.name} sum of mask {el.mask1} and {el.mask2}"

        raise ValueError(f"Unknown type {type(el)}")

    def pretty_print(self) -> str:
        return f"Calcualation Plan: {self.name}\n" + self._pretty_print(self.execution_tree, 0)

    def _pretty_print(self, elem: CalculationTree, indent) -> str:
        if isinstance(elem.operation, str):
            name = elem.operation
        else:
            name = self.get_el_name(elem.operation)
        if isinstance(elem.operation, (MeasurementCalculate, ROIExtractionProfile, MaskCreate)):
            if isinstance(elem.operation, ROIExtractionProfile):
                txt = elem.operation.pretty_print(analysis_algorithm_dict)
            else:
                txt = str(elem.operation)
            txt = "\n".join(txt.split("\n")[1:])
            name += "\n" + textwrap.indent(txt, " " * (indent + 4))

        if elem.children:
            suffix = "\n" + "\n".join(self._pretty_print(x, indent + 2) for x in elem.children)

        else:
            suffix = ""

        return " " * indent + name + suffix
