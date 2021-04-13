"""
This module contains utils for parallel batch calculation.
Main class is :py:class:`BatchManager` which is used to manage
parallel calculation

Main workflow is to add work with :py:meth:`BatchManager.add_work`
and consume results (:py:meth:`BatchManager.get_result`) until
:py:attr:`BatchManager.has_work` is evaluating to true

.. graphviz::

   digraph foo {
      "BatchManager" -> "BatchWorker"[arrowhead="crow"];
   }

"""
import logging
import multiprocessing
import os
import sys
import time
import traceback
import uuid
from enum import Enum
from queue import Empty, Queue
from threading import RLock, Timer
from typing import Any, Callable, Dict, List, Tuple

__author__ = "Grzegorz Bokota"


class SubprocessOrder(Enum):
    """
    Commands for process to put in queue
    """

    kill = 1
    wait = 2


class BatchManager:
    """
    This class is used for manage pending works.
    It use :py:class:`.BatchWorker` for running calculation.

    :type task_queue: Queue
    :type order_queue: Queue
    :type result_queue: Queue
    :type calculation_dict: dict
    :type process_list: list[multiprocessing.Process]
    """

    def __init__(self):
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.order_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.calculation_dict = self.manager.dict()
        self.number_off_available_process = 1
        self.number_off_process = 0
        self.number_off_alive_process = 0
        self.work_task = 0
        self.in_work = False
        self.process_list = []
        self.locker = RLock()

    def get_result(self) -> List[Tuple[uuid.UUID, Any]]:
        """
        Clean result queue and return it as list

        :return: List of results as tuple where first element is uuid of job and second is
            function result or tuple with exception as first argument and second is traceback
        """
        res = []
        while not self.result_queue.empty():
            res.append(self.result_queue.get())
        self.work_task -= len(res)
        if self.work_task == 0:
            logging.debug("computation finished")
            Timer(0.1, self._change_process_num, args=[-self.number_off_available_process]).start()
        return res

    def add_work(self, individual_parameters_list: List, global_parameters, fun: Callable[[Any, Any], Any]) -> str:
        """
        This function add next works to internal structures.
        Number of works is length of ``individual_parameters_list``

        :param individual_parameters_list: list of individual parameters for fun.
            For each element ``fun`` will be called with element as first argument
        :param global_parameters: second argument of fun. If has field uuid then it is used as work uuid
        :param fun: two argument function which will be used to run calculation.
            First argument is task specific, second is const for whole work.
        :return: work uuid
        """
        self.calculation_dict[global_parameters.uuid] = global_parameters, fun
        self.work_task += len(individual_parameters_list)
        if hasattr(global_parameters, "uuid"):
            task_uuid = global_parameters.uuid
        else:
            task_uuid = uuid.uuid4()
        for el in individual_parameters_list:
            self.task_queue.put((el, task_uuid))
        if self.number_off_available_process > self.number_off_process:
            for _ in range(self.number_off_available_process - self.number_off_process):
                self._spawn_process()
        self.in_work = True
        return task_uuid

    def _spawn_process(self):
        with self.locker:
            process = multiprocessing.Process(
                target=spawn_worker, args=(self.task_queue, self.order_queue, self.result_queue, self.calculation_dict)
            )
            process.start()
            self.process_list.append(process)
            self.number_off_alive_process += 1
            self.number_off_process += 1

    @property
    def has_work(self) -> bool:
        """Check if Manager has pending or processed work and if all results are consumed"""
        return self.work_task > 0 or (not self.result_queue.empty())

    def kill_jobs(self):
        for p in self.process_list:
            p.terminate()

    def set_number_of_process(self, num: int):
        """
        Change number of workers which should be used for calculation

        :param num: target number of process
        """
        process_diff = num - self.number_off_available_process
        logging.debug(f"[set_number_of_process] process diff: {process_diff}")
        self.number_off_available_process = num
        if not self.has_work:
            return
        self._change_process_num(process_diff)

    def _change_process_num(self, process_diff):
        if process_diff > 0:
            for _ in range(process_diff):
                self._spawn_process()
        else:
            for _ in range(-process_diff):
                logging.debug("[set_number_of_process] process kill")
                self.order_queue.put(SubprocessOrder.kill)
            with self.locker:
                self.number_off_process += process_diff
            self.join_all()

    def join_all(self):
        logging.debug(f"Join begin {len(self.process_list)} {self.number_off_process}")
        with self.locker:
            if len(self.process_list) > self.number_off_process:
                to_remove = []
                logging.debug(f"Process list start {self.process_list}")
                for p in self.process_list:
                    if not p.is_alive():
                        p.join()
                        self.number_off_alive_process -= 1
                        to_remove.append(p)
                for p in to_remove:
                    self.process_list.remove(p)
                self.number_off_alive_process -= len(to_remove)
                logging.debug(f"Process list end {self.process_list}")
            # FIXME self.number_off_alive_process,  self.number_off_process negative values
            if len(self.process_list) > self.number_off_process and len(self.process_list) > 0:
                logging.info(
                    "Wait on process, time {}, {}, {}, {}".format(
                        time.time(), self.number_off_alive_process, len(self.process_list), self.number_off_process
                    )
                )
                Timer(1, self.join_all).start()

    @property
    def finished(self):
        """Check if any process is running"""
        logging.debug(self.process_list)
        return len(self.process_list) == 0


class BatchWorker:
    """
    Worker spawned by :py:class:`BatchManager` instance

    :param task_queue: Queue with task data
    :param order_queue: Queue with additional orders (like kill)
    :param result_queue: Queue to put result
    :param calculation_dict: to store global parameters of task
    """

    def __init__(
        self,
        task_queue: Queue,
        order_queue: Queue,
        result_queue: Queue,
        calculation_dict: Dict[uuid.UUID, Tuple[Any, Callable[[Any, Any], Any]]],
    ):
        self.task_queue = task_queue
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.calculation_dict = calculation_dict

    def calculate_task(self, val: Tuple[Any, uuid.UUID]):
        """
        Calculate single task.
        ``val`` is tuple with two elements (task_data, uuid).
        function and global parameters are obtained from :py:attr:`.calculation_dict`
        """
        data, task_uuid = val
        global_data, fun = self.calculation_dict[task_uuid]
        try:
            res = fun(data, global_data)
            self.result_queue.put((task_uuid, res))
        except Exception as e:  # pragma: no cover
            traceback.print_exc()
            exc_type, _exc_obj, exc_tb = sys.exc_info()
            f_name = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, f_name, exc_tb.tb_lineno, file=sys.stderr)
            self.result_queue.put((task_uuid, (-1, [(e, traceback.extract_tb(e.__traceback__))])))

    def run(self):
        """Worker main loop"""
        logging.debug(f"Process started {os.getpid()}")
        while True:
            if not self.order_queue.empty():
                try:
                    order = self.order_queue.get_nowait()
                    logging.debug(f"Order message: {order}")
                    if order == SubprocessOrder.kill:
                        break
                except Empty:  # pragma: no cover
                    pass
            if not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    self.calculate_task(task)
                except Empty:
                    time.sleep(0.1)
                    continue
                except MemoryError:  # pragma: no cover
                    pass
                except OSError:  # pragma: no cover
                    pass
                except Exception as ex:  # pragma: no cover
                    logging.warning(f"Unsupported exception {ex}")
            else:
                time.sleep(0.1)
        logging.info(f"Process {os.getpid()} ended")


def spawn_worker(task_queue: Queue, order_queue: Queue, result_queue: Queue, calculation_dict: Dict[uuid.UUID, Any]):
    """
    Function for spawning worker. Designed as argument for :py:meth:`multiprocessing.Process`.

    :param task_queue: Queue with tasks
    :param order_queue: Queue with additional orders (like kill)
    :param result_queue: Queue for calculation result
    :param calculation_dict: dict with global parameters
    """
    worker = BatchWorker(task_queue, order_queue, result_queue, calculation_dict)
    worker.run()
