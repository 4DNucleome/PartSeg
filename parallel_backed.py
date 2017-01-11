import multiprocessing
from backend import SegmentationProfile, StatisticProfile
from collections import namedtuple
from enum import Enum
from uuid import uuid4
from Queue import Queue, Empty
import logging
import time
import os
from threading import Timer, RLock

__author__ = "Grzegorz Bokota"


class SubprocessOrder(Enum):
    kill = 1
    wait = 2


class BatchManager(object):
    """
    :type statistic_place_dict: dict[str,set[str]]
    :type task_queue: Queue
    :type order_queue: Queue
    :type result_queue: Queue
    :type calculation_dict: dict
    :type process_list: list[multiprocessing.Process]

    """
    def __init__(self):
        self.statistic_place_dict = dict()
        self.manager = multiprocessing.Manager()
        self.task_queue = self.manager.Queue()
        self.order_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.calculation_dict = self.manager.dict()
        self.number_off_process = 1
        self.number_off_alive_process = 0
        self.in_work = False
        self.process_list = []
        self.locker = RLock()
        pass

    def add_work(self, work_settings):
        for el in work_settings:
            self.task_queue.put(el)
        if self.number_off_process > self.number_off_alive_process:
            for i in range(self.number_off_process - self.number_off_alive_process):
                self.spawn_process()
        self.in_work = True

    def spawn_process(self):
        with self.locker:
            process = multiprocessing.Process(target=spawn_worker, args=(self.task_queue, self.order_queue,
                                                                         self.result_queue, self.calculation_dict))
            process.start()
            self.process_list.append(process)
            self.number_off_alive_process += 1

    @property
    def has_work(self):
        return self.in_work

    def set_number_off_process(self, num):
        process_diff = num - self.number_off_process
        self.number_off_process = num
        if not self.has_work:
            return
        if process_diff > 0:
            for i in range(process_diff):
                self.spawn_process()
        else:
            for i in range(-process_diff):
                self.order_queue.put(SubprocessOrder.kill)
            self.join_all()

    def is_sheet_name_use(self, file_path, name):
        if file_path not in self.statistic_place_dict:
            return False
        if name not in self.statistic_place_dict:
            return False
        return True

    def join_all(self):
        logging.debug("Join begin")
        with self.locker:
            if len(self.process_list) > self.number_off_process:
                to_remove = []
                logging.debug("Process list start {}".format(self.process_list))
                for p in self.process_list:
                    if not p.is_alive():
                        p.join()
                        self.number_off_alive_process -= 1
                        to_remove.append(p)
                for p in to_remove:
                    self.process_list.remove(p)
                self.number_off_alive_process -= len(to_remove)
                logging.debug("Process list end {}".format(self.process_list))
                if len(self.process_list) > self.number_off_process:
                    logging.debug("Wait on process, time {}".format(time.time()))
                    Timer(1, self.join_all, ()).start()

    @property
    def finished(self):
        return len(self.process_list) == 0

    def get_responses(self):
        res = []
        while not self.result_queue.empty():
            res.append(self.result_queue.get())
        return res


class BatchWorker(object):
    """
    :type task_queue: Queue
    :type order_queue: Queue
    :type result_queue: Queue
    :type calculation_dict: dict
    """
    def __init__(self, task_queue, order_queue, result_queue, calculation_dict):
        self.task_queue = task_queue
        self.order_queue = order_queue
        self.result_queue = result_queue
        self.calculation_dict = calculation_dict
        pass

    def calculate_task(self, task):
        self.result_queue.put(task)

    def run(self):
        logging.debug("Process started {}".format(os.getpid()))
        while True:
            if not self.order_queue.empty():
                try:
                    order = self.order_queue.get_nowait()
                    logging.debug("Order message: {}".format(order))
                    if order == SubprocessOrder.kill:
                        break
                except Empty:
                    pass
            if not self.task_queue.empty():
                try:
                    task = self.task_queue.get_nowait()
                    logging.debug("Task message: {}".format(task))
                    self.calculate_task(task)
                except Empty:
                    time.sleep(0.1)
                    continue
                except MemoryError:
                    pass
                except IOError:
                    pass
                except Exception as ex:
                    logging.warning("Unsupported exception {}".format(ex))
            else:
                time.sleep(0.1)
                continue
        logging.info("Process {} ended".format(os.getpid()))


def spawn_worker(task_queue, order_queue, result_queue, calculation_dict):
    worker = BatchWorker(task_queue, order_queue, result_queue, calculation_dict)
    worker.run()
    pass


