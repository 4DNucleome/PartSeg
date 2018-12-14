import multiprocessing
from enum import Enum
from queue import Queue, Empty
import logging
import time
import os
import sys
from threading import Timer, RLock
import traceback

__author__ = "Grzegorz Bokota"


class SubprocessOrder(Enum):
    kill = 1
    wait = 2


class Work(object):
    def __init__(self, task_list, ):
        self.task_list = task_list


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
        self.number_off_available_process = 1
        self.number_off_process = 0
        self.number_off_alive_process = 0
        self.work_task = 0
        self.in_work = False
        self.process_list = []
        self.locker = RLock()
        pass

    def get_result(self):
        res = []
        while not self.result_queue.empty():
            res.append(self.result_queue.get())
        self.work_task -= len(res)
        if self.work_task == 0:
            logging.debug("computation finished")
            Timer(0.1, self._change_process_num, (-self.number_off_available_process,)).start()
        return res

    def add_work(self, calc_settings, global_settings, function):
        self.calculation_dict[global_settings.uuid] = global_settings, function
        self.work_task += len(calc_settings)
        for el in calc_settings:
            self.task_queue.put((el, global_settings.uuid))
        if self.number_off_available_process > self.number_off_process:
            for i in range(self.number_off_available_process - self.number_off_process):
                self.spawn_process()
        self.in_work = True

    def spawn_process(self):
        with self.locker:
            process = multiprocessing.Process(target=spawn_worker, args=(self.task_queue, self.order_queue,
                                                                         self.result_queue, self.calculation_dict))
            process.start()
            self.process_list.append(process)
            self.number_off_alive_process += 1
            self.number_off_process += 1

    @property
    def has_work(self):
        return self.work_task > 0 or (not self.result_queue.empty())

    def set_number_off_process(self, num):
        process_diff = num - self.number_off_available_process
        logging.debug("[set_number_off_process] process diff: {}".format(process_diff))
        self.number_off_available_process = num
        if not self.has_work:
            return
        self._change_process_num(process_diff)

    def _change_process_num(self, process_diff):
        if process_diff > 0:
            for i in range(process_diff):
                self.spawn_process()
        else:
            for i in range(-process_diff):
                logging.debug("[set_number_off_process] process kill")
                self.order_queue.put(SubprocessOrder.kill)
            with self.locker:
                self.number_off_process += process_diff
            self.join_all()

    def is_sheet_name_use(self, file_path, name):
        if file_path not in self.statistic_place_dict:
            return False
        if name not in self.statistic_place_dict:
            return False
        return True

    def join_all(self):
        logging.debug("Join begin {} {}".format(len(self.process_list), self.number_off_process))
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
        logging.debug(self.process_list)
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

    def calculate_task(self, val):
        data, task_uuid = val
        global_data, function = self.calculation_dict[task_uuid]
        try:
            self.result_queue.put((task_uuid, function(data, global_data)))
        except Exception as e:
            traceback.print_exc()
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            self.result_queue.put((task_uuid, e))

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


