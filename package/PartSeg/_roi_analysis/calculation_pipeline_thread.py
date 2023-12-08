import typing

import numpy as np

from PartSeg.common_backend.progress_thread import ProgressTread
from PartSegCore.analysis.analysis_utils import SegmentationPipeline
from PartSegCore.analysis.calculate_pipeline import PipelineResult, calculate_pipeline
from PartSegImage import Image


class CalculatePipelineThread(ProgressTread):
    result: typing.Optional[PipelineResult]

    def __init__(self, image: Image, mask: typing.Union[np.ndarray, None], pipeline: SegmentationPipeline, parent=None):
        super().__init__(parent=parent)
        self.image = image
        self.mask = mask
        self.pipeline = pipeline
        self.result = None

    def run(self):
        try:
            self.result = calculate_pipeline(
                image=self.image, mask=self.mask, pipeline=self.pipeline, report_fun=self.info_function
            )
        except Exception as e:  # pylint: disable=broad-except
            self.error_signal.emit(e)
