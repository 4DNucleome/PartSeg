import logging
import os
import sys
import typing
import uuid
from abc import ABCMeta, abstractmethod
from collections import namedtuple
from copy import copy, deepcopy
from enum import Enum
from deprecation import deprecated
from six import add_metaclass

from PartSeg.utils.universal_const import Units
from ..analysis.save_register import save_dict
from ..analysis.statistics_calculation import StatisticProfile
from PartSeg.utils.algorithm_describe_base import SegmentationProfile
from ..mask_create import MaskProperty
from ..class_generator import BaseSerializableClass


class MaskBase:
    name: str

# MaskCreate = namedtuple("MaskCreate", ['name', 'radius'])

class MaskCreate(MaskBase, BaseSerializableClass):
    mask_property: MaskProperty

    def __str__(self):
        return f"Mask create: {self.name}\n" + str(self.mask_property).split("\n", 1)[1]

class MaskUse(MaskBase, BaseSerializableClass):
    pass

class MaskSum(MaskBase, BaseSerializableClass):
    mask1: str
    mask2: str

class MaskIntersection(MaskBase, BaseSerializableClass):
    mask1: str
    mask2: str

class SaveBase:
    suffix: str
    directory: str

class Save(BaseSerializableClass):
    suffix: str
    directory: str
    algorithm: str
    short_name: str
    values: dict


class StatisticCalculate(BaseSerializableClass):
    channel: int
    units: Units
    statistic_profile: StatisticProfile
    name_prefix: str

    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, channel: int, units: Units, statistic_profile: StatisticProfile, name_prefix: str): ...

    @property
    def name(self):
        return self.statistic_profile.name

    def __str__(self):
        channel = "Like segmentation" if self.channel == -1 else str(self.channel)
        desc = str(self.statistic_profile).split('\n', 1)[1]
        return  f"{self.__class__.name}\nChannel: {channel}\nUnits: {self.units}\n{desc}\n"






ChooseChanel = namedtuple("ChooseChanel", ["chanel_num"])

"""MaskCreate.__new__.__defaults__ = (0,)
CmapProfile.__new__.__defaults__ = (False,)
ProjectSave.__new__.__defaults__ = (False,)
MaskSave.__new__.__defaults__ = (False,)
XYZSave.__new__.__defaults__ = (False,)
ImageSave.__new__.__defaults__ = (False,)"""


def get_save_path(op: Save, calculation):
    """
    :type calculation: FileCalculation
    :param op: operation to do
    :param calculation: information about calculation
    :return: str
    """
    extension = save_dict[op.algorithm].get_default_extension()
    rel_path = os.path.relpath(calculation.file_path, calculation.base_prefix)
    rel_path, ext = os.path.splitext(rel_path)
    if op.directory:
        file_path = os.path.join(calculation.result_prefix, rel_path, op.suffix + extension)
    else:
        file_path = os.path.join(calculation.result_prefix, rel_path + op.suffix + extension)
    return file_path


class MaskMapper:
    name: str
    @abstractmethod
    def get_mask_path(self, file_path):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    def is_ready(self):
        return True


class MaskSuffix(MaskMapper, BaseSerializableClass):
    suffix: str

    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, name: str, suffix: str): ...

    def get_mask_path(self, file_path):
        base, ext = os.path.splitext(file_path)
        return base + self.suffix + ext

    def get_parameters(self):
        return {"name": self.name, "suffix": self.suffix}


class MaskSub(MaskMapper, BaseSerializableClass):
    base: str
    rep: str

    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, name: str, base: str, rep: str): ...

    def get_mask_path(self, file_path):
        dir_name, filename = os.path.split(file_path)
        filename = filename.replace(self.base, self.rep)
        return os.path.join(dir_name, filename)

    def get_parameters(self):
        return {"name": self.name, "base": self.base, "rep": self.rep}


class MaskFile(MaskMapper, BaseSerializableClass):
    path_to_file: str
    name_dict: typing.Optional[dict] = None

    # noinspection PyUnusedLocal
    # noinspection PyMissingConstructor
    def __init__(self, name: str, path_to_file: str, name_dict: typing.Optional[dict] = None):
        ...

    def is_ready(self):
        return os.path.exists(self.path_to_file)

    def get_mask_path(self, file_path):
        if self.name_dict is None:
            self.parse_map()
        try:
            return self.name_dict[os.path.normpath(file_path)]
        except KeyError:
            return None
        except AttributeError:
            return None

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
                    logging.error(
                        "Error in parsing map file\nline {}\n{}\nfrom file{}".format(i, line, self.path_to_file))
                    continue
                file_name = file_name.strip()
                mask_name = mask_name.strip()
                if not os.path.abspath(file_name):
                    file_name = os.path.normpath(os.path.join(dir_name, file_name))
                if not os.path.abspath(mask_name):
                    mask_name = os.path.normpath(os.path.join(dir_name, mask_name))
                self.name_dict[file_name] = mask_name


class Operations(Enum):
    reset_to_base = 1
    # leave_the_biggest = 2


class PlanChanges(Enum):
    add_node = 1
    remove_node = 2
    replace_node = 3


class CalculationTree:
    def __init__(self, operation: typing.Union[BaseSerializableClass, SegmentationProfile, StatisticCalculate, str],
                 children: typing.List['CalculationTree']):
        self.operation = operation
        self.children = children


class NodeType(Enum):
    segment = 1
    mask = 2
    statics = 3
    root = 4
    save = 5
    none = 6
    file_mask = 7
    channel_choose = 8


class Calculation(object):
    """
    :type file_list: list[str]
    :type base_prefix: str
    :type result_prefix: str
    :type statistic_file_path: str
    :type sheet_name: str
    :type calculation_plan: CalculationPlan

    """
    def __init__(self, file_list, base_prefix, result_prefix, statistic_file_path, sheet_name, calculation_plan,
                 voxel_size):
        self.file_list = file_list
        self.base_prefix = base_prefix
        self.result_prefix = result_prefix
        self.statistic_file_path = statistic_file_path
        self.sheet_name = sheet_name
        self.calculation_plan = calculation_plan
        self.uuid = uuid.uuid4()
        self.voxel_size = voxel_size


class FileCalculation(object):
    """
    :type file_path: st
    :type calculation: Calculation
    """
    def __init__(self, file_path, calculation):
        self.file_path = file_path
        self.calculation = calculation

    @property
    def base_prefix(self):
        return self.calculation.base_prefix

    @property
    def result_prefix(self):
        return self.calculation.result_prefix

    @property
    def calculation_plan(self):
        return self.calculation.calculation_plan

    @property
    def uuid(self):
        return self.calculation.uuid

    @property
    def voxel_size(self):
        return self.calculation.voxel_size


class CalculationPlan(object):
    """
    :type current_pos: list[int]
    :type name: str
    :type segmentation_count: int
    """
    correct_name = {MaskCreate.__name__: MaskCreate, MaskUse.__name__: MaskUse, Save.__name__: Save,
                    StatisticCalculate.__name__: StatisticCalculate, SegmentationProfile.__name__: SegmentationProfile,
                    MaskSuffix.__name__: MaskSuffix, MaskSub.__name__: MaskSub, MaskFile.__name__: MaskFile,
                    Operations.__name__: Operations, ChooseChanel.__name__: ChooseChanel,
                    MaskIntersection.__name__: MaskIntersection, MaskSum.__name__: MaskSum}

    def __init__(self, tree: typing.Optional[CalculationTree] = None, name:str = ""):
        if tree is None:
            self.execution_tree = CalculationTree("root", [])
        else:
            self.execution_tree = tree
        self.segmentation_count = 0
        self.name = name
        self.current_pos = []
        self.changes = []
        self.current_node = None

    def get_statistics(self, node=None):
        """
        :type node: CalculationTree
        :param node:
        :return: list[StatisticCalculate]
        """
        if node is None:
            node = self.execution_tree
        if isinstance(node.operation, StatisticCalculate):
            return [node.operation]
        else:
            res = []
            for el in node.children:
                res.extend(self.get_statistics(el))
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
        self.execution_tree = CalculationTree("root", [])
        self.current_pos = []

    def __copy__(self):
        return CalculationPlan(name=self.name, tree=deepcopy(self.execution_tree))

    def __deepcopy__(self):
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
        if isinstance(node.operation, MaskCreate) or isinstance(node.operation, MaskMapper):
            res.add(node.operation.name)
        for el in node.children:
            res |= self.get_mask_names(el)
        return res

    def get_file_mask_names(self):
        node = self.get_node()
        used_mask = self.get_reused_mask()
        tree_mask_names = self.get_mask_names(node)
        return used_mask & tree_mask_names, used_mask

    def _get_reused_mask(self, node):
        """
        :type node: CalculationTree
        :param node:
        :return:
        """
        used_mask = set()
        for el in node.children:
            if isinstance(el.operation, MaskUse):
                used_mask.add(el.operation.name)
            elif isinstance(el.operation, MaskSum):
                used_mask.add(el.operation.mask1)
                used_mask.add(el.operation.mask2)
            elif isinstance(el.operation, MaskIntersection):
                used_mask.add(el.operation.mask1)
                used_mask.add(el.operation.mask2)
            elif isinstance(el.operation, ChooseChanel):
                used_mask |= self._get_reused_mask(el)
        return used_mask

    def get_reused_mask(self) -> set:
        return self._get_reused_mask(self.execution_tree)

    def get_node_type(self)-> NodeType:
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
        if isinstance(node.operation, StatisticCalculate):
            return NodeType.statics
        if isinstance(node.operation, SegmentationProfile):
            return NodeType.segment
        if isinstance(node.operation, Save):
            return NodeType.save
        if isinstance(node.operation, ChooseChanel):
            return NodeType.channel_choose
        if isinstance(node.operation, MaskUse):
            return NodeType.file_mask
        if isinstance(node.operation, Operations):
            if node.operation == Operations.reset_to_base:
                return NodeType.mask
        raise ValueError("[get_node_type] unknown node type {}".format(node.operation))

    def add_step(self, step):
        if self.current_pos is None:
            return
        node = self.get_node()
        node.children.append(CalculationTree(step, []))
        if isinstance(step, SegmentationProfile):
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
        if len(node.children) > 0:
            return True
        return False

    def remove_step(self):
        path = copy(self.current_pos)
        pos = path[-1]
        parent_node = self.get_node(path[:-1])
        del parent_node.children[pos]
        self.changes.append((self.current_pos, None, PlanChanges.remove_node))
        self.current_pos = self.current_pos[:-1]

    def is_segmentation(self):
        return self.segmentation_count > 0

    def set_name(self, text):
        self.name = text

    def get_parameters(self):
        return self.dict_dump()

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
        else:
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
        mask_mapper_list = []
        for el in self.execution_tree.children:
            if isinstance(el.operation, MaskMapper):
                mask_mapper_list.append(el.operation)
        return mask_mapper_list

    def set_path_to_mapping_file(self, num, path):
        for el in self.execution_tree.children:
            if num == 0:
                el.operation.path_to_file = path
                return el.operation
            if isinstance(el.operation, MaskFile):
                num -= 1

    @deprecated
    def recursive_dump(self, node, pos):
        """
        :type node: CalculationTree
        :type pos: list[int]
        :param node:
        :param pos:
        :return: list[(list[int], object, PlanChanges)]
        """
        sub_dict = dict()
        el = node.operation
        sub_dict["type"] = el.__class__.__name__
        if issubclass(el.__class__, tuple):
            # noinspection PyProtectedMember
            sub_dict["values"] = el._asdict()
        elif isinstance(el, StatisticProfile):
            sub_dict["values"] = el.get_parameters()
        elif isinstance(el, SegmentationProfile):
            sub_dict["values"] = el.get_parameters()
        elif isinstance(el, MaskMapper):
            sub_dict["values"] = el.get_parameters()
        elif isinstance(el, Operations):
            sub_dict["values"] = {"value": el.value}
        else:
            raise ValueError("Not supported type {}".format(el))
        res = [(pos, sub_dict, PlanChanges.add_node.value)]
        for i, el in enumerate(node.children):
            res.extend(self.recursive_dump(el, pos + [i]))
        return res

    @deprecated
    def dict_dump(self):
        res = dict()
        res["name"] = self.name
        flat_tree = []
        for i, x in enumerate(self.execution_tree.children):
            flat_tree.extend(self.recursive_dump(x, [i]))
        res["execution_tree"] = flat_tree
        return res

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
            print(el, file=sys.stderr)
            raise ValueError("Unknown type {}".format(el.__class__.__name__))
        if isinstance(el, Operations):
            if el == Operations.reset_to_base:
                return "reset project to base image with mask"
        if isinstance(el, ChooseChanel):
            return "Chose chanel: chanel num {}".format(el.chanel_num)
        if isinstance(el, SegmentationProfile):
            """if el.leave_biggest:
                return "Segmentation: {} (only biggest)".format(el.name)"""
            return "Segmentation: {}".format(el.name)
        if isinstance(el, StatisticCalculate):
            if el.name_prefix == "":
                return "Statistics: {}".format(el.name)
            else:
                return "Statistics: {} with prefix: {}".format(el.name, el.name_prefix)
        if isinstance(el, MaskCreate):
            if el.name != "":
                return "Create mask: {}".format(el.name)
            else:
                return "Create mask:"
        if isinstance(el, MaskUse):
            return "Use mask: {}".format(el.name)
        if isinstance(el, MaskSuffix):
            return "File mask: {} with suffix {}".format(el.name, el.suffix)
        if isinstance(el, MaskSub):
            return "File mask: {} substitution {} on {}".format(el.name, el.base, el.rep)
        if isinstance(el, MaskFile):
            return "File mapping mask: {}, {}".format(el.name, el.path_to_file)
        if isinstance(el, Save):
            base = el.short_name
            if el.directory:
                text = f"Save {base} in directory with name " + el.suffix
            else:
                if el.suffix != "":
                    text = base + " with suffix " + el.suffix
                else:
                    text = base
            return text
        if isinstance(el, MaskIntersection):
            if el.name == "":
                return "Mask intersection of mask {} and {}".format(el.mask1, el.mask2)
            else:
                return "Mask {} intersection of mask {} and {}".format(el.name, el.mask1, el.mask2)
        if isinstance(el, MaskSum):
            if el.name == "":
                return "Mask sum of mask {} and {}".format(el.mask1, el.mask2)
            else:
                return "Mask {} sum of mask {} and {}".format(el.name, el.mask1, el.mask2)

        raise ValueError("Unknown type {}".format(type(el)))
