from parallel_backed import BatchManager
from backend import Settings, MaskChange
from  statistics_calculation import StatisticProfile
from segment import Segment, SegmentationProfile
from io_functions import load_project, save_to_project, save_to_cmap
from os import path
import tifffile
import pandas as pd
from calculation_plan import CalculationPlan, CalculationTree, MaskMapper, MaskUse, MaskCreate, ProjectSave, \
    CmapProfile, Operations, ChooseChanel, FileCalculation
from copy import copy
from utils import dict_set_class
from queue import Queue
from collections import Counter, OrderedDict, defaultdict


def do_calculation(file_path, calculation):
    """
    :type file_path: str
    :type calculation: Calculation
    :param calculation:
    :return:
    """
    calc = CalculationProcess()
    return calc.do_calculation(FileCalculation(file_path, calculation))


class CalculationProcess(object):
    def __init__(self):
        self.settings = Settings(None)
        self.segment = Segment(self.settings, callback=False)
        self.reused_mask = set()
        self.mask_dict = dict()
        self.calculation = None
        self.statistics = []

    def do_calculation(self, calculation):
        """
        :type calculation: FileCalculation
        :param calculation:
        :return:
        """
        self.calculation = calculation
        self.reused_mask = calculation.calculation_plan.get_reused_mask()
        self.mask_dict = {}
        self.statistics = []
        print(calculation.file_path)
        ext = path.splitext(calculation.file_path)[1]
        if ext in [".tiff", ".tif", ".lsm"]:
            image = tifffile.imread(calculation.file_path)
            self.settings.image = image
        elif ext in [".tgz", "gz", ".tbz2", ".bz2"]:
            load_project(calculation.file_path, self.settings, self.segment)
        else:
            raise ValueError("Unknown file type: {} {}". format(ext, calculation.file_path))

        self.iterate_over(calculation.calculation_plan.execution_tree)
        return calculation.file_path, self.statistics

    def iterate_over(self, node):
        """
        :type node: CalculationTree
        :param node:
        :return:
        """
        for el in node.children:
            self.recursive_calculation(el)

    def recursive_calculation(self, node):
        """
        :type node: CalculationTree
        :param node:
        :return:
        """
        if isinstance(node.operation, MaskMapper):
            mask = tifffile.imread(node.operation.get_mask_path(self.settings.file_path))
            self.settings.mask = mask
            self.iterate_over(node)
        elif isinstance(node.operation, SegmentationProfile):
            old_segment = copy(self.segment)
            current_profile = SegmentationProfile("curr", **self.settings.get_profile_dict())
            dict_set_class(self.settings,  node.operation.get_parameters(),
                           *SegmentationProfile.SEGMENTATION_PARAMETERS)
            self.segment.recalculate()
            self.iterate_over(node)
            dict_set_class(self.settings, current_profile.get_parameters(),
                           *SegmentationProfile.SEGMENTATION_PARAMETERS)
            self.segment = old_segment
        elif isinstance(node.operation, MaskUse):
            old_mask = self.settings.mask
            mask = self.mask_dict[node.operation.name]
            self.settings.mask = mask
            self.iterate_over(node)
            self.settings.mask = old_mask
        elif isinstance(node.operation, MaskCreate):
            if node.operation.name in self.reused_mask:
                mask = self.segment.get_segmentation()
                self.mask_dict[node.operation.name] = mask
            self.settings.change_segmentation_mask(self.segment, MaskChange.next_seg, False)
            self.iterate_over(node)
            self.settings.change_segmentation_mask(self.segment, MaskChange.prev_seg, False)
        elif isinstance(node.operation, ProjectSave):
            file_path = self.settings.file_path
            file_path = path.relpath(file_path, self.calculation.base_prefix)
            rel_path, ext = path.splitext(file_path)
            file_path = path.join(self.calculation.result_prefix, rel_path + node.operation.suffix+".tgz")
            save_to_project(file_path, self.settings, self.segment)
        elif isinstance(node.operation, CmapProfile):
            file_path = self.settings.file_path
            file_path = path.relpath(file_path, self.calculation.base_prefix)
            rel_path, ext = path.splitext(file_path)
            file_path = path.join(self.calculation.result_prefix, rel_path + node.operation.suffix + ".cmap")
            save_to_cmap(file_path, self.settings, self.segment, node.operation.gauss_type, False,
                         node.operation.center_data, rotate=node.operation.rotation_axis,
                         with_cutting=node.operation.cut_obsolete_area)
        elif isinstance(node.operation, Operations):
            if node.operation == Operations.segment_from_project:
                load_project(self.settings.file_path, self.settings, self.segment)
        elif isinstance(node.operation, ChooseChanel):
            image = self.settings.image
            new_image = image.take(node.operation.chanel_num, axis=node.operation.chanel_position)
            self.settings.add_image(new_image, self.calculation.file_path)
            self.iterate_over(node)
            self.settings.add_image(image, self.calculation.file_path)
        elif isinstance(node.operation, StatisticProfile):
            statistics = node.operation.calculate(self.settings.image, self.settings.gauss_image,
                                                  self.segment.get_segmentation(), self.segment.get_full_segmentation(),
                                                  self.settings.mask, self.settings.voxel_size)
            self.statistics.append(statistics)


class CalculationManager(object):
    def __init__(self):
        super(CalculationManager, self).__init__()
        self.batch_manager = BatchManager()
        self.calculation_queue = Queue()
        self.calculation_dict = OrderedDict()
        self.calculation_sizes = []
        self.calculation_size = 0
        self.calculation_done = 0
        self.counter_dict = OrderedDict()
        self.errors_list = []
        self.sheet_name = defaultdict(set)

    def is_valid_sheet_name(self, excel_path, sheet_name):
        return sheet_name not in self.sheet_name[excel_path]

    def add_calculation(self, calculation):
        """
        :type calculation: Calculation
        :param calculation: calculation
        :return:
        """
        self.sheet_name[calculation.statistic_file_path].add(calculation.sheet_name)
        self.calculation_dict[calculation.uuid] = calculation, calculation.calculation_plan.get_statistics()
        self.counter_dict[calculation.uuid] = 0
        size = len(calculation.file_list)
        self.calculation_sizes.append(size)
        self.calculation_size += size
        self.batch_manager.add_work(calculation.file_list, calculation, do_calculation)

    @property
    def has_work(self):
        return self.batch_manager.has_work

    def set_number_of_workers(self, val):
        print("Number off process {}".format(val))
        self.batch_manager.set_number_off_process(val)

    def get_results(self):
        responses = self.batch_manager.get_result()
        new_errors = []
        for uuid, el in responses:
            self.calculation_done += 1
            self.counter_dict[uuid] += 1
            if isinstance(el, Exception):
                self.errors_list.append(el)
                new_errors.append(el)
            else:
                with open(self.calculation_dict[uuid][0].statistic_file_path, 'a') as ff:
                    ff.write(str(el)+"\n")
        return new_errors, self.calculation_done, zip(self.counter_dict.values(), self.calculation_sizes)






