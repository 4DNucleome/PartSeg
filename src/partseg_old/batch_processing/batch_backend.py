import logging
import threading
from collections import OrderedDict, defaultdict
from copy import copy
from enum import Enum
from os import path
from queue import Queue

import numpy as np
import pandas as pd
import tifffile

from partseg_old.backend import Settings, MaskChange
from partseg_old.batch_processing.calculation_plan import CalculationTree, MaskMapper, MaskUse, MaskCreate, ProjectSave, \
    ImageSave, XYZSave, \
    CmapProfile, Operations, ChooseChanel, FileCalculation, MaskIntersection, MaskSum, MaskSave, get_save_path
from partseg_old.batch_processing.parallel_backed import BatchManager
from partseg_old.io_functions import load_project, save_to_project, save_to_cmap, save_to_xyz
from partseg_old.segment import Segment, SegmentationProfile
from partseg_old.statistics_calculation import StatisticProfile
from partseg_utils import dict_set_class


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
        self.settings.voxel_size = calculation.voxel_size
        ext = path.splitext(calculation.file_path)[1]
        if ext in [".tiff", ".tif", ".lsm"]:
            image = tifffile.imread(calculation.file_path)
            self.settings.image = image
            self.settings.file_path = calculation.file_path
        elif ext in [".tgz", ".gz", ".tbz2", ".bz2"]:
            load_project(calculation.file_path, self.settings, self.segment)
        else:
            raise ValueError("Unknown file type: {} {}". format(ext, calculation.file_path))
        self.iterate_over(calculation.calculation_plan.execution_tree)
        return path.relpath(calculation.file_path, calculation.base_prefix), self.statistics

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
            mask = tifffile.imread(node.operation.get_mask_path(self.calculation.file_path))
            mask = (mask > 0).astype(np.uint8)
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
        elif isinstance(node.operation, MaskSum):
            old_mask = self.settings.mask
            mask1 = self.mask_dict[node.operation.mask1]
            mask2 = self.mask_dict[node.operation.mask2]
            mask = np.logical_or(mask1, mask2).astype(np.uint8)
            self.settings.mask = mask
            self.iterate_over(node)
            self.settings.mask = old_mask
        elif isinstance(node.operation, MaskIntersection):
            old_mask = self.settings.mask
            mask1 = self.mask_dict[node.operation.mask1]
            mask2 = self.mask_dict[node.operation.mask2]
            mask = np.logical_and(mask1, mask2).astype(np.uint8)
            self.settings.mask = mask
            self.iterate_over(node)
            self.settings.mask = old_mask
        elif isinstance(node.operation, MaskSave):
            if self.settings.mask is None:
                return
            file_path = get_save_path(node.operation, self.calculation)
            tifffile.imsave(file_path, self.settings.mask)
        elif isinstance(node.operation, ImageSave):
            file_path = get_save_path(node.operation, self.calculation)
            tifffile.imsave(file_path, self.settings.image)
        elif isinstance(node.operation, XYZSave):
            file_path = get_save_path(node.operation, self.calculation)
            save_to_xyz(file_path, self.settings, self.segment)
        elif isinstance(node.operation, MaskCreate):
            if node.operation.name in self.reused_mask:
                mask = self.segment.get_segmentation()
                self.mask_dict[node.operation.name] = mask
            self.settings.mask_dilate_radius = node.operation.radius
            self.settings.change_segmentation_mask(self.segment, MaskChange.next_seg, False)
            self.iterate_over(node)
            self.settings.change_segmentation_mask(self.segment, MaskChange.prev_seg, False)
        elif isinstance(node.operation, ProjectSave):
            file_path = get_save_path(node.operation, self.calculation)
            save_to_project(file_path, self.settings, self.segment)
        elif isinstance(node.operation, CmapProfile):
            file_path = get_save_path(node.operation, self.calculation)
            save_to_cmap(file_path, self.settings, self.segment, node.operation.gauss_type, False,
                         node.operation.center_data, rotate=node.operation.rotation_axis,
                         with_cutting=node.operation.cut_obsolete_area)
        elif isinstance(node.operation, Operations):
            if node.operation == Operations.segment_from_project:
                load_project(self.settings.file_path, self.settings, self.segment)
                self.iterate_over(node)
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
        else:
            logging.error("Unknown operation {} {}".format(type(node.operation), node.operation))


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
        self.writer = DataWriter()

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
        self.writer.add_data_part(calculation)

    @property
    def has_work(self):
        return self.batch_manager.has_work

    def set_number_of_workers(self, val):
        logging.debug("Number off process {}".format(val))
        self.batch_manager.set_number_off_process(val)

    def get_results(self):
        responses = self.batch_manager.get_result()
        new_errors = []
        for uuid, el in responses:
            self.calculation_done += 1
            self.counter_dict[uuid] += 1
            calculation = self.calculation_dict[uuid][0]
            if isinstance(el, Exception):
                self.errors_list.append(el)
                new_errors.append(el)
            else:
                errors = self.writer.add_result(el, calculation)
                for err in errors:
                    new_errors.append(err)
            if self.counter_dict[uuid] == len(calculation.file_list):
                errors = self.writer.calculation_finished(calculation)
                for err in errors:
                    new_errors.append(err)
        return new_errors, self.calculation_done, zip(self.counter_dict.values(), self.calculation_sizes)


class FileType(Enum):
    excel_xlsx_file = 1
    excel_xls_file = 2
    text_file = 3


class SheetData(object):
    def __init__(self, name, columns):
        self.name = name
        self.columns = ["name"] + columns
        self.data_frame = pd.DataFrame([], columns=self.columns)
        self.row_list = []

    def add_data(self, data):
        self.row_list.append(data)

    def add_data_list(self, data):
        self.row_list.extend(data)

    def get_data_to_write(self):
        df = pd.DataFrame(self.row_list, columns=self.columns)
        df2 = self.data_frame.append(df)
        self.data_frame = df2.reset_index(drop=True)
        self.row_list = []
        return self.name, self.data_frame


class FileData(object):
    component_str = "_components_"

    def __init__(self, calculation):
        """
        :type calculation: Calculation
        :param calculation:
        """
        self.file_path = calculation.statistic_file_path
        ext = path.splitext(calculation.statistic_file_path)[1]
        if ext == ".xlsx":
            self.file_type = FileType.excel_xlsx_file
        elif ext == ".xls":
            self.file_type = FileType.excel_xls_file
        else:
            self.file_type = FileType.text_file
        self.sheet_dict = dict()
        self.sheet_set = set()
        self.new_count = 0
        self.write_threshold = 40
        self.wrote_queue = Queue()
        self.error_queue = Queue()
        self.write_thread = threading.Thread(target=self.wrote_data_to_file)
        self.write_thread.daemon = True
        self.write_thread.start()
        self.add_data_part(calculation)

    def good_sheet_name(self, name):
        if self.file_type == FileType.text_file:
            return False, "Text file allow store only one sheet"
        if FileData.component_str in name:
            return False, "Sequence '{}' is reserved for auto generated sheets".format(FileData.component_str)
        if name in self.sheet_set:
            return False, "Sheet name {} already in use".format(name)
        return True, True

    def add_data_part(self, calculation):
        """
        :type calculation: Calculation
        :param calculation:
        :return:
        """
        if calculation.statistic_file_path != self.file_path:
            raise ValueError("[FileData] different file path {} vs {}".format(calculation.statistic_file_path,
                                                                              self.file_path))
        if calculation.sheet_name in self.sheet_set:
            raise ValueError("[FileData] sheet name {} already in use".format(calculation.sheet_name))
        statistics = calculation.calculation_plan.get_statistics()
        component_information = [x.get_component_info() for x in statistics]
        num = 1
        sheet_list = []
        header_list = []
        main_header = []
        for i, el in enumerate(component_information):
            local_header = []
            if any([x[1] for x in el]):
                sheet_list.append("{}{}{} - {}".format(calculation.sheet_name, FileData.component_str, num,
                                                       statistics[i].name_prefix+statistics[i].name))
                num += 1
            else:
                sheet_list.append(None)
            for name, comp in el:
                if comp:
                    local_header.append(name)
                else:
                    main_header.append(name)
            header_list.append(local_header)

        self.sheet_dict[calculation.uuid] = (
            SheetData(calculation.sheet_name, main_header),
            [SheetData(name, header_list[i]) if name is not None else None for i, name in enumerate(sheet_list)],
            component_information)

    def wrote_data(self, uuid, data):
        self.new_count += 1
        main_sheet, component_sheets, component_information = self.sheet_dict[uuid]
        name = data[0]
        data_list = [name]
        for el, comp_sheet, comp_info in zip(data[1], component_sheets, component_information):
            comp_list = []
            for val, info in zip(el.values(), comp_info):
                if info[1]:
                    comp_list.append(val)
                else:
                    data_list.append(val)
            if len(comp_list) > 0:
                comp_list.insert(0, ["{}_comp_{}".format(name, i) for i in range(len(comp_list[0]))])
                comp_list = zip(*comp_list)
            if comp_sheet is not None:
                comp_sheet.add_data_list(comp_list)
        main_sheet.add_data(data_list)
        if self.new_count >= self.write_threshold:
            self.dump_data()
            self.new_count = 0

    def dump_data(self):
        data = []
        for main_sheet, component_sheets, _ in self.sheet_dict.values():
            data.append(main_sheet.get_data_to_write())
            for sheet in component_sheets:
                if sheet is not None:
                    data.append(sheet.get_data_to_write())
        self.wrote_queue.put(data)

    def wrote_data_to_file(self):
        while True:
            data = self.wrote_queue.get()
            if data == "finish":
                break
            try:
                if self.file_type == FileType.text_file:
                    base_path, ext = path.splitext(self.file_path)
                    for sheet_name, data_frame in data:
                        data_frame.to_csv(base_path + "_" + sheet_name + ext)
                else:
                    writer = pd.ExcelWriter(self.file_path)
                    for sheet_name, data_frame in data:
                        data_frame.to_excel(writer, sheet_name=sheet_name)
                    writer.save()
            except Exception as e:
                logging.error(e)
                self.error_queue.put(e)

    def get_errors(self):
        res = []
        while not self.error_queue.empty():
            res.append(self.error_queue.get())
        return res

    def finish(self):
        self.wrote_queue.put("finish")

    def is_empty_sheet(self, sheet_name):
        return sheet_name not in self.sheet_set


class DataWriter(object):
    def __init__(self):
        self.file_dict = dict()

    def is_empty_sheet(self, file_path, sheet_name):
        if FileData.component_str in sheet_name:
            return False
        if file_path not in self.file_dict:
            return True
        return self.file_dict[file_path].is_empty_sheet(sheet_name)

    def add_data_part(self, calculation):
        if calculation.statistic_file_path in self.file_dict:
            self.file_dict[calculation.statistic_file_path].add_data_part(calculation)
        else:
            self.file_dict[calculation.statistic_file_path] = FileData(calculation)

    def add_result(self, data, calculation):
        if calculation.statistic_file_path not in self.file_dict:
            raise ValueError("Unknown statistic file")
        file_writer = self.file_dict[calculation.statistic_file_path]
        file_writer.wrote_data(calculation.uuid, data)
        return file_writer.get_errors()

    def finish(self):
        for file_data in self.file_dict.keys():
            file_data.finish()

    def calculation_finished(self, calculation):
        if calculation.statistic_file_path not in self.file_dict:
            raise ValueError("Unknown statistic file")
        self.file_dict[calculation.statistic_file_path].dump_data()
        return self.file_dict[calculation.statistic_file_path].get_errors()
