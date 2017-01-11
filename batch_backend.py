from parallel_backed import BatchManager
from backend import CalculationPlan, Settings, Segment
from os import path
import tifffile
import pandas as pd


def do_calculation(calculation_plan, file_path):
    """
    :type calculation_plan: CalculationPlan
    :type file_path: str
    :param calculation_plan:
    :param file_path:
    :return:
    """
    settings = Settings(None)
    segment = Segment(settings)
    ext = path.split(file_path)[1]
    if ext in [".tiff", ".tif", ".lsm"]:
        image = tifffile.imread(file_path)

    pass


class CalculationManager(object):
    def __init__(self):
        super(CalculationManager, self).__init__()
        self.batch_manager = BatchManager()



