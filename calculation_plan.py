from abc import ABCMeta, abstractmethod
from six import add_metaclass
import os
from copy import copy
from collections import namedtuple
import logging
from enum import Enum
from statistics_calculation import StatisticProfile
from segment import SegmentationProfile

MaskCreate = namedtuple("MaskCreate", ['name', 'radius'])
MaskUse = namedtuple("MaskUse", ['name'])
CmapProfile = namedtuple("CmapProfile", ["suffix", "gauss_type", "center_data", "rotation_axis", "cut_obsolete_are"])
ProjectSave = namedtuple("ProjectSave", ["suffix"])
ChooseChanel = namedtuple("ChooseChanel", ["chanel_position", "chanel_num"])

MaskCreate.__new__.__defaults__ = (0,)


@add_metaclass(ABCMeta)
class MaskMapper(object):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def get_mask_path(self, file_path):
        pass

    @abstractmethod
    def get_parameters(self):
        pass


class MaskSuffix(MaskMapper):
    def __init__(self, name, suffix):
        super(MaskSuffix, self).__init__(name)
        self.suffix = suffix

    def get_mask_path(self, file_path):
        base, ext = os.path.splitext(file_path)
        return base + self.suffix + ext

    def get_parameters(self):
        return {"name": self.name, "suffix": self.suffix}


class MaskSub(MaskMapper):
    def __init__(self, name, base, rep):
        super(MaskSub, self).__init__(name)
        self.base = base
        self.rep = rep

    def get_mask_path(self, file_path):
        dir_name, filename = os.path.split(file_path)
        filename = filename.replace(self.base, self.rep)
        return os.path.join(dir_name, filename)

    def get_parameters(self):
        return {"name": self.name, "base": self.base}


class MaskFile(MaskMapper):
    def __init__(self, name, path_to_file):
        super(MaskFile, self).__init__(name)
        self.path_to_file = path_to_file
        self.name_dict = None

    def get_mask_path(self, file_path):
        if self.name_dict is None:
            self.parse_map()
        return self.name_dict[os.path.normpath(file_path)]

    def get_parameters(self):
        return {"name": self.name, "path_to_file": self.path_to_file}

    def set_map_path(self, value):
        self.path_to_file = value

    def parse_map(self, sep=";"):
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
    segment_from_project = 1


class PlanChanges(Enum):
    add_node = 1
    remove_node = 2
    replace_node = 3


class CalculationTree(object):
    def __init__(self, operation, children):
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


class CalculationPlan(object):
    """
    :type current_pos: list[int]
    :type name: str
    :type segmentation_count: int
    """
    correct_name = {MaskCreate.__name__: MaskCreate, MaskUse.__name__: MaskUse, CmapProfile.__name__: CmapProfile,
                    StatisticProfile.__name__: StatisticProfile, SegmentationProfile.__name__: SegmentationProfile,
                    MaskSuffix.__name__: MaskSuffix, MaskSub.__name__: MaskSub, MaskFile.__name__: MaskFile,
                    ProjectSave.__name__: ProjectSave, Operations.__name__: Operations,
                    ChooseChanel.__name__: ChooseChanel}

    def __init__(self):
        self.execution_list = []
        self.execution_tree = CalculationTree("root", [])
        self.segmentation_count = 0
        self.name = ""
        self.current_pos = []
        self.changes = []

    def get_changes(self):
        ret = self.changes
        self.changes = []
        return ret

    def position(self):
        return self.current_pos

    def set_position(self, value):
        self.current_pos = value

    def clean(self):
        self.execution_list = []
        self.execution_tree = CalculationTree("root", [])
        self.current_pos = []

    def get_node(self, search_pos=None):
        """
        :param search_pos:
        :return: CalculationTree
        """
        node = self.execution_tree
        if search_pos is None:
            search_pos = self.current_pos
        for pos in search_pos:
            node = node.children[pos]
        return node

    def _get_mask_name(self, node):
        """
        :type node: CalculationTree
        :param node:
        :return: set[str]
        """
        res = set()
        if isinstance(node.operation, MaskCreate) or isinstance(node.operation, MaskMapper):
            res.add(node.operation.name)
        for el in node.children:
            res |= self._get_mask_name(el)
        return res

    def get_mask_names(self):
        node = self.get_node()
        used_mask = set()
        for el in self.execution_tree.children:
            if isinstance(el.operation, MaskUse):
                used_mask.add(el.operation.name)
        tree_mask_names = self._get_mask_name(node)
        return used_mask & tree_mask_names, used_mask

    def get_node_type(self):
        if self.current_pos is None:
            return NodeType.none
        if not self.current_pos:
            return NodeType.root
        # print("Pos {}".format(self.current_pos))
        node = self.get_node()
        if isinstance(node.operation, MaskMapper):
            return NodeType.file_mask
        if isinstance(node.operation, MaskCreate):
            return NodeType.mask
        if isinstance(node.operation, StatisticProfile):
            return NodeType.statics
        if isinstance(node.operation, SegmentationProfile):
            return NodeType.segment
        if isinstance(node.operation, ProjectSave) or isinstance(node.operation, CmapProfile):
            return NodeType.save
        if isinstance(node.operation, ChooseChanel):
            return NodeType.root
        if isinstance(node.operation, MaskUse):
            return NodeType.file_mask
        logging.error("[get_node_type] unknown node type {}".format(node.operation))

    def add_step(self, step):
        if self.current_pos is None:
            return
        node = self.get_node()
        self.execution_list.append(step)
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
        node.operation.name = name
        self.changes.append((self.current_pos, node, PlanChanges.replace_node))

    def __len__(self):
        return len(self.execution_list)

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

    def pop(self):
        el = self.execution_list.pop()
        if isinstance(el, SegmentationProfile):
            self.segmentation_count -= 1
        self.execution_tree = None
        return el

    def is_segmentation(self):
        return self.segmentation_count > 0

    def set_name(self, text):
        self.name = text

    def get_parameters(self):
        return self.dict_dump()

    def get_execution_tree(self):
        return self.execution_tree

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
            sub_dict["values"] = el.__dict__
        elif isinstance(el, StatisticProfile):
            sub_dict["values"] = el.get_parameters()
        elif isinstance(el, SegmentationProfile):
            sub_dict["values"] = el.get_parameters()
        elif isinstance(el, MaskMapper):
            sub_dict["values"] = el.get_parameters()
        else:
            raise ValueError("Not supported type {}".format(el))
        res = [(pos, sub_dict, PlanChanges.add_node.value)]
        for i, el in enumerate(node.children):
            res.extend(self.recursive_dump(el, pos + [i]))
        return res

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
            res_plan.add_step(CalculationPlan.correct_name[el["type"]](**el["values"]))
        res_plan.changes = []
        return res_plan

    @staticmethod
    def get_el_name(el):
        """
        :param el: Plan element
        :return: str
        """
        if el.__class__.__name__ not in CalculationPlan.correct_name.keys():
            print(el)
            raise ValueError("Unknown type {}".format(el.__class__.__name__))
        if isinstance(el, Operations):
            if el == Operations.segment_from_project:
                return "Segment from project"
        if isinstance(el, ChooseChanel):
            return "Chose chanel, chanel pos: {}, chanel num {}".format(el.chanel_position, el.chanel_num)
        if isinstance(el, SegmentationProfile):
            return "Segmentation: {}".format(el.name)
        if isinstance(el, StatisticProfile):
            if el.name_prefix == "":
                return "Statistics: {}".format(el.name)
            else:
                return "Statistics: {} with prefix: {}".format(el.name, el.name_prefix)
        if isinstance(el, MaskCreate):
            if el.name != "":
                return "Create mask: {}, dilate radius: {}".format(el.name, el.radius)
            else:
                return "Create mask with dilate radius: {}".format(el.radius)
        if isinstance(el, MaskUse):
            return "Use mask: {}".format(el.name)
        if isinstance(el, CmapProfile):
            if el.suffix == "":
                return "Camp save"
            else:
                return "Cmap save with suffix: {}".format(el.suffix)
        if isinstance(el, MaskSuffix):
            return "File mask: {} with suffix {}".format(el.name, el.suffix)
        if isinstance(el, MaskSub):
            return "File mask: {} substitution {} on {}".format(el.name, el.base, el.rep)
        if isinstance(el, MaskFile):
            return "File mapping mask: {}".format(el.name)
        if isinstance(el, ProjectSave):
            if el.suffix != "":
                return "Save to project with suffix {}".format(el.suffix)
            else:
                return "Save to project"

        raise ValueError("Unknown type {}".format(type(el)))
