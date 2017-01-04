import multiprocessing
from qt_import import QTimer
from backend import Profile, StatisticProfile
from collections import namedtuple
from enum import Enum
__author__ = "Grzegorz Bokota"


class BatchManager(object):
    def __init__(self):
        pass


class BatchWorker(object):
    def __init__(self):
        pass

MaskCreate = namedtuple("MaskCreate", ['name'])
MaskUse = namedtuple("MaskUse", ['name'])
CmapProfile = namedtuple("CmapProfile", ["suffix", "gauss_type", "center_data", "rotation_axis", "cut_obsolete_are"])


class Operations(Enum):
    clean_mask = 1


class CalculationPlan(object):
    # TODO Batch plans dump and load
    def __init__(self):
        self.execution_list = []
        self.execution_tree = None
        self.segmentation_count = 0
        self.name = ""

    def clean(self):
        self.execution_list = []

    def add_step(self, step):
        self.execution_list.append(step)
        if isinstance(step, Profile):
            self.segmentation_count += 1

    def __len__(self):
        return len(self.execution_list)

    def pop(self):
        el = self.execution_list.pop()
        if isinstance(el, Profile):
            self.segmentation_count -= 1

    def is_segmentation(self):
        return self.segmentation_count > 0

    def build_execution_tree(self):
        pass

    def set_name(self, text):
        self.name = text

