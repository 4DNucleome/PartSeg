"""
This module contains PartSeg function used for calculate in batch processing

Calculation hierarchy:

.. graphviz::

   digraph calc {
      rankdir="LR";
      "CalculationManager"[shape=rectangle style=filled];
      "BatchManager"[shape=rectangle];
      "BatchWorker"[shape=rectangle];
      "CalculationManager" -> "BatchManager" -> "BatchWorker"[arrowhead="crow"];

      "CalculationManager" -> "DataWriter"[arrowhead="inv"];
      "DataWriter"[shape=rectangle];
      "FileData"[shape=rectangle];
      "SheetData"[shape=rectangle];
      "DataWriter" -> "FileData" -> "SheetData"[arrowhead="crow"];
   }

"""
import json
import logging
import os
import threading
import traceback
import uuid
from collections import OrderedDict
from enum import Enum
from os import path
from queue import Queue
from traceback import StackSummary
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
import SimpleITK
import tifffile
import xlsxwriter

from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.calculation_plan import (
    BaseCalculation,
    Calculation,
    CalculationPlan,
    CalculationTree,
    FileCalculation,
    MaskCreate,
    MaskIntersection,
    MaskMapper,
    MaskSum,
    MaskUse,
    MeasurementCalculate,
    Operations,
    RootType,
    Save,
    get_save_path,
)
from PartSegCore.analysis.io_utils import ProjectTuple
from PartSegCore.analysis.load_functions import LoadMaskSegmentation, LoadProject, load_dict
from PartSegCore.analysis.measurement_base import AreaType, PerComponent
from PartSegCore.analysis.measurement_calculation import MeasurementResult
from PartSegCore.analysis.save_functions import save_dict
from PartSegCore.mask_create import calculate_mask
from PartSegCore.segmentation.algorithm_base import SegmentationAlgorithm, report_empty_fun
from PartSegImage import Image, TiffImageReader

from ...io_utils import WrongFileTypeException
from ...project_info import AdditionalLayerDescription, HistoryElement
from ...roi_info import ROIInfo
from ...segmentation import RestartableAlgorithm
from .. import PartEncoder
from .parallel_backend import BatchManager


class ResponseData(NamedTuple):
    path_to_file: str
    values: List[MeasurementResult]


CalculationResultList = List[ResponseData]
ErrorInfo = Tuple[Exception, Union[StackSummary, Tuple[Dict, StackSummary]]]
WrappedResult = Tuple[int, List[Union[ErrorInfo, ResponseData]]]


def prepare_error_data(exception: Exception) -> ErrorInfo:
    try:
        from sentry_sdk.serializer import serialize
        from sentry_sdk.utils import event_from_exception

        event = event_from_exception(exception)[0]
        event = serialize(event)
        return exception, (event, traceback.extract_tb(exception.__traceback__))
    except ImportError:  # pragma: no cover
        return exception, traceback.extract_tb(exception.__traceback__)


def do_calculation(file_info: Tuple[int, str], calculation: BaseCalculation) -> WrappedResult:
    """
    Main function which will be used for run calculation.
    It create :py:class:`.CalculationProcess` and call it method
    :py:meth:`.CalculationProcess.do_calculation`

    :param file_info: index and path to file which should be processed
    :param calculation: calculation description
    """
    SimpleITK.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    calc = CalculationProcess()
    index, file_path = file_info
    try:
        return index, calc.do_calculation(FileCalculation(file_path, calculation))
    except Exception as e:
        return index, [prepare_error_data(e)]


class CalculationProcess:
    """
    Main class to calculate PartSeg calculation plan.
    To support other operations overwrite :py:meth:`recursive_calculation`
    call super function to support already defined operations.
    """

    def __init__(self):
        self.reused_mask = set()
        self.mask_dict = {}
        self.calculation = None
        self.measurement: List[MeasurementResult] = []
        self.image: Optional[Image] = None
        self.roi_info: Optional[ROIInfo] = None
        self.additional_layers: Dict[str, AdditionalLayerDescription] = {}
        self.mask: Optional[np.ndarray] = None
        self.history: List[HistoryElement] = []
        self.algorithm_parameters: dict = {}
        self.results: CalculationResultList = []

    def do_calculation(self, calculation: FileCalculation) -> CalculationResultList:
        """
        Main function for calculation process

        :param calculation: calculation to do.
        :return:
        """
        self.calculation = calculation
        self.reused_mask = calculation.calculation_plan.get_reused_mask()
        self.mask_dict = {}
        self.measurement = []
        self.results = []
        operation = calculation.calculation_plan.execution_tree.operation
        ext = path.splitext(calculation.file_path)[1]
        metadata = {"default_spacing": calculation.voxel_size}
        if operation == RootType.Image:
            for load_class in load_dict.values():
                if load_class.partial() or load_class.number_of_files() != 1:
                    continue
                if ext in load_class.get_extensions():
                    projects = load_class.load([calculation.file_path], metadata=metadata)
                    break
            else:  # pragma: no cover
                raise ValueError("File type not supported")
        elif operation == RootType.Project:
            projects = LoadProject.load([calculation.file_path], metadata=metadata)
        else:  # operation == RootType.Mask_project
            try:
                projects = LoadProject.load([calculation.file_path], metadata=metadata)
            except (KeyError, WrongFileTypeException):
                # TODO identify exceptions
                projects = LoadMaskSegmentation.load([calculation.file_path], metadata=metadata)

        if isinstance(projects, ProjectTuple):
            projects = [projects]
        for project in projects:
            project: ProjectTuple
            self.image = project.image
            if operation == RootType.Mask_project:
                self.mask = project.mask
            if operation == RootType.Project:
                self.mask = project.mask
                # FIXME when load annotation from project is done
                self.roi_info = project.roi_info
                self.additional_layers = project.additional_layers
                self.history = project.history
                self.algorithm_parameters = project.algorithm_parameters

            self.iterate_over(calculation.calculation_plan.execution_tree)
            for el in self.measurement:
                el.set_filename(path.relpath(project.image.file_path, calculation.base_prefix))
            self.results.append(
                ResponseData(path.relpath(project.image.file_path, calculation.base_prefix), self.measurement)
            )
            self.measurement = []
        return self.results

    def iterate_over(self, node: Union[CalculationTree, List[CalculationTree]]):
        """
        Execute calculation on node children or list oof nodes

        :type node: CalculationTree
        :param node:
        :return:
        """
        if isinstance(node, CalculationTree):
            node = node.children
        for el in node:
            self.recursive_calculation(el)

    def step_load_mask(self, operation: MaskMapper, children: List[CalculationTree]):
        """
        Load mask using mask mapper (mask can be defined with suffix, substitution, or file with mapping saved,
        then iterate over ``children`` nodes.

        :param MaskMapper operation: operation to perform
        :param List[CalculationTree] children: list of nodes to iterate over with applied mask
        """
        mask_path = operation.get_mask_path(self.calculation.file_path)
        if mask_path == "":  # pragma: no cover
            raise ValueError("Empty path to mask.")
        if not os.path.exists(mask_path):
            raise OSError(f"Mask file {mask_path} does not exists")
        with tifffile.TiffFile(mask_path) as mask_file:
            mask = mask_file.asarray()
            mask = TiffImageReader.update_array_shape(mask, mask_file.series[0].axes)[..., 0]
        mask = (mask > 0).astype(np.uint8)
        try:
            mask = self.image.fit_array_to_image(mask)[0]
            # TODO fix this time bug fix
        except ValueError:  # pragma: no cover
            raise ValueError("Mask do not fit to given image")
        old_mask = self.mask
        self.mask = mask
        self.iterate_over(children)
        self.mask = old_mask

    def step_segmentation(self, operation: ROIExtractionProfile, children: List[CalculationTree]):
        """
        Perform segmentation and iterate over ``children`` nodes

        :param ROIExtractionProfile operation: Specification of segmentation operation
        :param List[CalculationTree] children: list of nodes to iterate over after perform segmentation
        """
        segmentation_class = analysis_algorithm_dict.get(operation.algorithm, None)
        if segmentation_class is None:  # pragma: no cover
            raise ValueError(f"Segmentation class {operation.algorithm} do not found")
        segmentation_algorithm: RestartableAlgorithm = segmentation_class()
        segmentation_algorithm.set_image(self.image)
        segmentation_algorithm.set_mask(self.mask)
        segmentation_algorithm.set_parameters(**operation.values)
        result = segmentation_algorithm.calculation_run(report_empty_fun)
        backup_data = self.roi_info, self.additional_layers, self.algorithm_parameters
        self.roi_info = ROIInfo(result.roi, result.roi_annotation, result.alternative_representation)
        self.additional_layers = result.additional_layers
        self.algorithm_parameters = {"algorithm_name": operation.algorithm, "values": operation.values}
        self.iterate_over(children)
        self.roi_info, self.additional_layers, self.algorithm_parameters = backup_data

    def step_mask_use(self, operation: MaskUse, children: List[CalculationTree]):
        """
        use already defined mask and iterate over ``children`` nodes

        :param MaskUse operation:
        :param List[CalculationTree] children: list of nodes to iterate over after perform segmentation
        """
        old_mask = self.mask
        mask = self.mask_dict[operation.name]
        self.mask = mask
        self.iterate_over(children)
        self.mask = old_mask

    def step_mask_operation(self, operation: Union[MaskSum, MaskIntersection], children: List[CalculationTree]):
        """
        Generate new mask by sum or intersection of existing and iterate over ``children`` nodes

        :param operation: mask operation to perform
        :type operation: Union[MaskSum, MaskIntersection]
        :param List[CalculationTree] children: list of nodes to iterate over after perform segmentation
        """
        old_mask = self.mask
        mask1 = self.mask_dict[operation.mask1]
        mask2 = self.mask_dict[operation.mask2]
        if isinstance(operation, MaskSum):
            mask = np.logical_or(mask1, mask2).astype(np.uint8)
        else:
            mask = np.logical_and(mask1, mask2).astype(np.uint8)
        self.mask = mask
        self.iterate_over(children)
        self.mask = old_mask

    def step_save(self, operation: Save):
        """
        Perform save operation selected in plan.

        :param Save operation: save definition
        """
        save_class = save_dict[operation.algorithm]
        project_tuple = ProjectTuple(
            file_path="",
            image=self.image,
            roi_info=self.roi_info,
            additional_layers=self.additional_layers,
            mask=self.mask,
            history=self.history,
            algorithm_parameters=self.algorithm_parameters,
        )
        save_path = get_save_path(operation, self.calculation)
        save_class.save(save_path, project_tuple, operation.values)

    def step_mask_create(self, operation: MaskCreate, children: List[CalculationTree]):
        """
        Create mask from current segmentation state using definition

        :param MaskCreate operation: mask create description.
        :param List[CalculationTree] children: list of nodes to iterate over after perform segmentation
        """
        mask = calculate_mask(
            mask_description=operation.mask_property,
            roi=self.roi_info.roi,
            old_mask=self.mask,
            spacing=self.image.spacing,
            time_axis=self.image.time_pos,
        )
        if operation.name in self.reused_mask:
            self.mask_dict[operation.name] = mask
        history_element = HistoryElement.create(
            self.roi_info,
            self.mask,
            self.algorithm_parameters,
            operation.mask_property,
        )
        backup = self.mask, self.history
        self.mask = mask
        self.history.append(history_element)
        self.iterate_over(children)
        self.mask, self.history = backup

    def step_measurement(self, operation: MeasurementCalculate):
        """
        Calculate measurement defined in current operation.

        :param MeasurementCalculate operation: definition of measurement to calculate
        """
        channel = operation.channel
        if channel == -1:
            segmentation_class: Type[SegmentationAlgorithm] = analysis_algorithm_dict.get(
                self.algorithm_parameters["algorithm_name"], None
            )
            if segmentation_class is None:  # pragma: no cover
                raise ValueError(f"Segmentation class {self.algorithm_parameters['algorithm_name']} do not found")
            channel = self.algorithm_parameters["values"][segmentation_class.get_channel_parameter_name()]

        # FIXME use additional information
        old_mask = self.image.mask
        self.image.set_mask(self.mask)
        measurement = operation.measurement_profile.calculate(
            self.image,
            channel,
            self.roi_info,
            operation.units,
        )
        self.measurement.append(measurement)
        self.image.set_mask(old_mask)

    def recursive_calculation(self, node: CalculationTree):
        """
        Identify node type and then call proper ``step_*`` function

        :param CalculationTree node: Node to be proceed
        """
        if isinstance(node.operation, MaskMapper):
            self.step_load_mask(node.operation, node.children)
        elif isinstance(node.operation, ROIExtractionProfile):
            self.step_segmentation(node.operation, node.children)
        elif isinstance(node.operation, MaskUse):
            self.step_mask_use(node.operation, node.children)
        elif isinstance(node.operation, (MaskSum, MaskIntersection)):
            self.step_mask_operation(node.operation, node.children)
        elif isinstance(node.operation, Save):
            self.step_save(node.operation)
        elif isinstance(node.operation, MaskCreate):
            self.step_mask_create(node.operation, node.children)
        elif isinstance(node.operation, Operations):  # pragma: no cover
            # backward compatibility
            self.iterate_over(node)
        elif isinstance(node.operation, MeasurementCalculate):
            self.step_measurement(node.operation)
        else:  # pragma: no cover
            raise ValueError(f"Unknown operation {type(node.operation)} {node.operation}")


class BatchResultDescription(NamedTuple):
    """
    Tuple to handle information about part of calculation result.
    """

    errors: List[Tuple[str, ErrorInfo]]  #: list of errors occurred during calculation
    global_counter: int  #: total number of calculated steps
    jobs_status: Iterable[Tuple[int, int]]  #: for each job information about progress


class CalculationInfo(NamedTuple):
    calculation: Calculation
    measurement: List[MeasurementCalculate]


class CalculationManager:
    """
    This class manage batch processing in PartSeg.

    """

    def __init__(self):
        self.batch_manager = BatchManager()
        self.calculation_queue = Queue()
        self.calculation_dict: Dict[uuid.UUID, CalculationInfo] = OrderedDict()
        self.calculation_sizes = []
        self.calculation_size = 0
        self.calculation_done = 0
        self.counter_dict = OrderedDict()
        self.errors_list = []
        self.writer = DataWriter()

    def is_valid_sheet_name(self, excel_path: str, sheet_name: str) -> bool:
        """
        Check if sheet name can be used

        :param str excel_path: path which allow identify excel file
        :param str sheet_name: name of excel sheet
        :return:
        :rtype: bool
        """
        return self.writer.is_empty_sheet(excel_path, sheet_name)

    def add_calculation(self, calculation: Calculation):
        """
        :param calculation: Calculation
        """
        self.calculation_dict[calculation.uuid] = CalculationInfo(
            calculation, calculation.calculation_plan.get_measurements()
        )
        self.counter_dict[calculation.uuid] = 0
        size = len(calculation.file_list)
        self.calculation_sizes.append(size)
        self.calculation_size += size
        self.batch_manager.add_work(
            list(enumerate(calculation.file_list)), calculation.get_base_calculation(), do_calculation
        )
        self.writer.add_data_part(calculation)

    @property
    def has_work(self) -> bool:
        """
        Is still some calculation or data writing in progress
        """
        return self.batch_manager.has_work or not self.writer.writing_finished()

    def kill_jobs(self):
        self.batch_manager.kill_jobs()

    def set_number_of_workers(self, val: int):
        """
        Set number of workers to perform calculation.

        :param int val: number of workers.
        """
        logging.debug(f"Number off process {val}")
        self.batch_manager.set_number_of_process(val)

    def get_results(self) -> BatchResultDescription:
        """
        Consume results from :py:class:`BatchWorker` and transfer it to :py:class:`DataWriter`

        :return: information about calculation status
        :rtype: BatchResultDescription
        """
        responses: List[Tuple[uuid.UUID, WrappedResult]] = self.batch_manager.get_result()
        new_errors: List[Tuple[str, ErrorInfo]] = []
        for uuid_id, (ind, result_list) in responses:
            self.calculation_done += 1
            self.counter_dict[uuid_id] += 1
            calculation = self.calculation_dict[uuid_id].calculation
            for el in result_list:
                if isinstance(el, ResponseData):
                    errors = self.writer.add_result(el, calculation, ind=ind)
                    for err in errors:
                        new_errors.append((el.path_to_file, err))
                else:
                    file_info = calculation.file_list[ind] if ind != -1 else "unknown file"
                    self.writer.add_calculation_error(calculation, file_info, el[0])
                    self.errors_list.append((file_info, el))
                    new_errors.append((file_info, el))

                if self.counter_dict[uuid_id] == len(calculation.file_list):
                    errors = self.writer.calculation_finished(calculation)
                    for err in errors:
                        new_errors.append(("", err))
        return BatchResultDescription(
            new_errors, self.calculation_done, zip(self.counter_dict.values(), self.calculation_sizes)
        )


class FileType(Enum):
    excel_xlsx_file = 1
    excel_xls_file = 2
    text_file = 3


class SheetData:
    """
    Store single sheet information
    """

    def __init__(self, name: str, columns: List[Tuple[str, str]]):
        self.name = name
        self.columns = pd.MultiIndex.from_tuples([("name", "units")] + columns)
        self.data_frame = pd.DataFrame([], columns=self.columns)
        self.row_list: List[Any] = []

    def add_data(self, data, ind):
        if ind is None:
            ind = len(self.row_list)
        self.row_list.append((ind, data))

    def add_data_list(self, data, ind):
        if ind is None:
            ind = len(self.row_list)
        self.row_list.extend([(ind, x) for x in data])

    def get_data_to_write(self) -> Tuple[str, pd.DataFrame]:
        """
        Get data for write

        :return: sheet name and data to write
        :rtype: Tuple[str, pd.DataFrame]
        """
        sorted_row = [x[1] for x in sorted(self.row_list)]
        df = pd.DataFrame(sorted_row, columns=self.columns)
        df2 = self.data_frame.append(df)
        self.data_frame = df2.reset_index(drop=True)
        self.row_list = []
        return self.name, self.data_frame


class FileData:
    """
    Handle information about single file.

    This class run separate thread for writing purpose.
    This need additional synchronisation. but not freeze

    :param BaseCalculation calculation: calculation information
    :param int write_threshold: every how many lines of data are written to disk
    :cvar component_str: separator for per component sheet information
    """

    component_str = "_comp_"  #: separator for per component sheet information

    def __init__(self, calculation: BaseCalculation, write_threshold: int = 40):
        """
        :param BaseCalculation calculation: calculation information
        :param int write_threshold: every how many lines of data are written to disk
        """
        self.file_path = calculation.measurement_file_path
        ext = path.splitext(calculation.measurement_file_path)[1]
        if ext == ".xlsx":
            self.file_type = FileType.excel_xlsx_file
        elif ext == ".xls":  # pragma: no cover
            self.file_type = FileType.excel_xls_file
        else:  # pragma: no cover
            self.file_type = FileType.text_file
        self.writing = False
        data = SheetData("calculation_info", [("Description", "str"), ("JSON", "str")])
        data.add_data([str(calculation.calculation_plan), json.dumps(calculation.calculation_plan, cls=PartEncoder)], 0)
        self.sheet_dict = {}
        self.calculation_info = {}
        self.sheet_set = {"Errors"}
        self.new_count = 0
        self.write_threshold = write_threshold
        self.wrote_queue = Queue()
        self.error_queue = Queue()
        self.write_thread = threading.Thread(target=self.wrote_data_to_file)
        self.write_thread.daemon = True
        self.write_thread.start()
        self._error_info = []
        self.add_data_part(calculation)

    def finished(self):
        """check if any data wait on write to disc"""
        return not self.writing and self.wrote_queue.empty()

    def good_sheet_name(self, name: str) -> Tuple[bool, str]:
        """
        Check if sheet name can be used in current file.
        Return False if:

        * file is text file
        * contains :py:attr:`component_str` in name
        * name is already in use

        :param str name: sheet name
        :return: if can be used and error message
        """
        if self.file_type == FileType.text_file:
            return False, "Text file allow store only one sheet"
        if self.component_str in name:
            return False, f"Sequence '{FileData.component_str}' is reserved for auto generated sheets"
        if name in self.sheet_set:
            return False, f"Sheet name {name} already in use"
        return True, ""

    def add_data_part(self, calculation: BaseCalculation):
        """
        Add new calculation which result will be stored in handled file.

        :param BaseCalculation calculation: information about calculation
        :raises ValueError: when :py:attr:`measurement_file_path` is different to handled file
            or :py:attr:`sheet_name` name already is in use.
        """
        if calculation.measurement_file_path != self.file_path:
            raise ValueError(f"[FileData] different file path {calculation.measurement_file_path} vs {self.file_path}")
        if calculation.sheet_name in self.sheet_set:  # pragma: no cover
            raise ValueError(f"[FileData] sheet name {calculation.sheet_name} already in use")
        measurement = calculation.calculation_plan.get_measurements()
        component_information = [x.measurement_profile.get_component_info(x.units) for x in measurement]
        num = 1
        sheet_list = []
        header_list = []
        main_header = []
        for i, el in enumerate(component_information):
            local_header = []
            component_seg = False
            component_mask = False
            for per_component, area in measurement[i].measurement_profile.get_component_and_area_info():
                if per_component == PerComponent.Yes and area == AreaType.ROI:
                    component_seg = True
                if per_component == PerComponent.Yes and area != AreaType.ROI:
                    component_mask = True
            if component_seg:
                local_header.append(("Segmentation component", "num"))
            if component_mask:
                local_header.append(("Mask component", "num"))
            if any(x[1] for x in el):
                sheet_list.append(
                    "{}{}{} - {}".format(
                        calculation.sheet_name,
                        FileData.component_str,
                        num,
                        measurement[i].name_prefix + measurement[i].name,
                    )
                )
                num += 1
            else:
                sheet_list.append(None)
            for name, comp in el:
                local_header.append(name)
                if not comp:
                    main_header.append(name)
            header_list.append(local_header)

        self.sheet_dict[calculation.uuid] = (
            SheetData(calculation.sheet_name, main_header),
            [SheetData(name, header_list[i]) if name is not None else None for i, name in enumerate(sheet_list)],
            component_information,
        )
        self.calculation_info[calculation.uuid] = calculation.calculation_plan

    def wrote_data(self, uuid_id: uuid.UUID, data: ResponseData, ind: Optional[int] = None):
        """
        Add information to be stored in output file

        :param uuid.UUID uuid_id: calculation identifier
        :param ResponseData data: calculation result
        :param Optional[int] ind: element index
        """
        self.new_count += 1
        main_sheet, component_sheets, _component_information = self.sheet_dict[uuid_id]
        name = data.path_to_file
        data_list = [name]
        for el, comp_sheet in zip(data.values, component_sheets):
            data_list.extend(el.get_global_parameters()[1:])
            comp_list = el.get_separated()
            if comp_sheet is not None:
                comp_sheet.add_data_list(comp_list, ind)
        main_sheet.add_data(data_list, ind)
        if self.new_count >= self.write_threshold:
            self.dump_data()
            self.new_count = 0

    def wrote_errors(self, file_path, error_description):
        self.new_count += 1
        self._error_info.append((file_path, str(error_description)))

    def dump_data(self):
        """
        Fire writing data to disc
        """
        data = []
        for main_sheet, component_sheets, _ in self.sheet_dict.values():
            data.append(main_sheet.get_data_to_write())
            for sheet in component_sheets:
                if sheet is not None:
                    data.append(sheet.get_data_to_write())
        segmentation_info = [x for x in self.calculation_info.values()]
        self.wrote_queue.put((data, segmentation_info, self._error_info[:]))

    def wrote_data_to_file(self):
        """
        Main function to write data to hard drive.
        It is executed in separate thread.
        """
        while True:
            data = self.wrote_queue.get()
            if data == "finish":
                break
            self.writing = True
            try:
                if self.file_type == FileType.text_file:
                    base_path, ext = path.splitext(self.file_path)
                    for sheet_name, data_frame in data[0]:
                        data_frame.to_csv(base_path + "_" + sheet_name + ext)
                    continue
                file_path = self.file_path
                i = 0
                while i < 100:
                    i += 1
                    try:
                        self.write_to_excel(file_path, data)
                        break
                    except (PermissionError, OSError):
                        base, ext = path.splitext(self.file_path)
                        file_path = f"{base}({i}){ext}"
                if i == 100:  # pragma: no cover
                    raise PermissionError(f"Fail to write result excel {self.file_path}")
            except Exception as e:  # pragma: no cover
                logging.error(f"[batch_backend] {e}")
                self.error_queue.put(prepare_error_data(e))
            finally:
                self.writing = False

    @classmethod
    def write_to_excel(
        cls, file_path: str, data: Tuple[List[Tuple[str, pd.DataFrame]], List[CalculationPlan], List[Tuple[str, str]]]
    ):
        with pd.ExcelWriter(file_path) as writer:
            new_sheet_names = []
            ind = 0
            sheets, plans, errors = data
            for sheet_name, _ in sheets:
                if len(sheet_name) < 32:
                    new_sheet_names.append(sheet_name)
                else:
                    new_sheet_names.append(sheet_name[:27] + f"_{ind}_")
                    ind += 1
            for sheet_name, (_, data_frame) in zip(new_sheet_names, sheets):
                data_frame.to_excel(writer, sheet_name=sheet_name)
                sheet = writer.book.sheetnames[sheet_name]
                sheet.set_column(1, 1, 10)
                for i, (text, _unit) in enumerate(data_frame.columns[1:], start=2):
                    sheet.set_column(i, i, len(text) + 1)

            for calculation_plan in plans:
                cls.write_calculation_plan(writer, calculation_plan)

            if errors:
                errors_data = pd.DataFrame(errors, columns=["File path", "error description"])
                errors_data.to_excel(writer, "Errors")

    @staticmethod
    def write_calculation_plan(writer: pd.ExcelWriter, calculation_plan: CalculationPlan):
        book: xlsxwriter.Workbook = writer.book
        sheet_base_name = f"info {calculation_plan.name}"[:30]
        sheet_name = sheet_base_name
        if sheet_name in book.sheetnames:  # pragma: no cover
            for i in range(100):
                sheet_name = f"{sheet_base_name[:26]} ({i})"
                if sheet_name not in book.sheetnames:
                    break
            else:
                raise ValueError(f"Name collision in sheets with information about calculation plan: {sheet_name}")

        sheet = book.add_worksheet(sheet_name)
        cell_format = book.add_format({"bold": True})
        sheet.write("A1", "Plan Description", cell_format)
        sheet.write("B1", "Plan JSON", cell_format)
        description = calculation_plan.pretty_print()
        sheet.write("A2", description)
        sheet.set_row(1, description.count("\n") * 12 + 10)
        sheet.set_column(0, 0, max(map(len, description.split("\n"))))
        sheet.set_column(1, 1, 15)
        sheet.write("B2", json.dumps(calculation_plan, cls=PartEncoder, indent=2))

    def get_errors(self) -> List[ErrorInfo]:
        """
        Get list of errors occurred in last write
        """
        res = []
        while not self.error_queue.empty():
            res.append(self.error_queue.get())
        return res

    def finish(self):
        self.wrote_queue.put("finish")

    def is_empty_sheet(self, sheet_name) -> bool:
        return self.good_sheet_name(sheet_name)[0]


class DataWriter:
    """
    Handle information
    """

    def __init__(self):
        self.file_dict: Dict[str, FileData] = dict()

    def is_empty_sheet(self, file_path: str, sheet_name: str) -> bool:
        """
        Check if given pair of `file_path` and `sheet_name` can be used.

        :param str file_path: path to file to store measurement result
        :param str sheet_name: Name of excel sheet in which data will be stored
        :return: If calling :py:meth:`FileData.add_data_part` finish without error.
        :rtype: bool
        """
        if FileData.component_str in sheet_name:
            return False
        if file_path not in self.file_dict:
            return True
        return self.file_dict[file_path].is_empty_sheet(sheet_name)

    def add_data_part(self, calculation: BaseCalculation):
        """
        Add information about calculation
        """
        if calculation.measurement_file_path in self.file_dict:
            self.file_dict[calculation.measurement_file_path].add_data_part(calculation)
        else:
            self.file_dict[calculation.measurement_file_path] = FileData(calculation)

    def add_result(
        self, data: ResponseData, calculation: BaseCalculation, ind: Optional[int] = None
    ) -> List[ErrorInfo]:
        """
        Add calculation result to file writer

        :raises ValueError: when calculation.measurement_file_path is not added with :py:meth:`.add_data_part`
        """
        if calculation.measurement_file_path not in self.file_dict:
            raise ValueError("Unknown measurement file")
        file_writer = self.file_dict[calculation.measurement_file_path]
        file_writer.wrote_data(calculation.uuid, data, ind)
        return file_writer.get_errors()

    def add_calculation_error(self, calculation: BaseCalculation, file_path: str, error):
        file_writer = self.file_dict[calculation.measurement_file_path]
        file_writer.wrote_errors(file_path, error)

    def writing_finished(self) -> bool:
        """check if all data are written to disc"""
        return all(x.finished() for x in self.file_dict.values())

    def finish(self):
        """close all files"""
        for file_data in self.file_dict.values():
            file_data.finish()

    def calculation_finished(self, calculation) -> List[ErrorInfo]:
        """
        Force write data for given calculation.

        :raises ValueError: when measurement is not added with :py:meth:`.add_data_part`
        :return: list of errors during write.
        """
        if calculation.measurement_file_path not in self.file_dict:
            raise ValueError("Unknown measurement file")
        self.file_dict[calculation.measurement_file_path].dump_data()
        return self.file_dict[calculation.measurement_file_path].get_errors()
