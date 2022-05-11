import dataclasses
import sys

from qtpy.QtCore import QMutex, QThread, Signal

from PartSegCore.segmentation.algorithm_base import ROIExtractionAlgorithm, ROIExtractionResult


class SegmentationThread(QThread):
    """
    Method to run calculation task in separated Thread. This allows to not freeze main window.
    To get info if calculation is done connect to :py:meth:`~.QThread.finished`.
    """

    execution_done = Signal(ROIExtractionResult)
    """
    Signal contains result of segmentation algorithm. Emitted if calculation ends without exception and
    :py:meth:`SegmentationAlgorithm.calculation_run` return not None result.
    """
    progress_signal = Signal(str, int)
    """
    Signal with information about progress. This is proxy for :py:meth:`SegmentationAlgorithm.calculation_run`
    `report_fun` parameter`
    """
    info_signal = Signal(str)
    exception_occurred = Signal(Exception)
    """Signal emitted when some exception occur during calculation. """

    def __init__(self, algorithm: ROIExtractionAlgorithm):
        super().__init__()
        self.finished.connect(self.finished_task)
        self.algorithm = algorithm
        self.clean_later = False
        self.cache = None
        self._image = None
        self._mask = None
        self.mutex = QMutex()
        self.rerun = False, QThread.InheritPriority

    def get_info_text(self):
        """Proxy for :py:meth:`.SegmentationAlgorithm.get_info_text`."""
        return self.algorithm.get_info_text()

    def send_info(self, text, num):
        self.progress_signal.emit(text, num)

    def run(self):
        """the calculation are done here"""
        if self.algorithm.image is None:
            # assertion for running algorithm without image
            print(f"No image in class {self.algorithm.__class__}", file=sys.stderr)
            return
        try:
            segment_data = self.algorithm.calculation_run_wrap(self.send_info)
            if segment_data is not None:
                segment_data = dataclasses.replace(segment_data, file_path=self.algorithm.image.file_path)
        except Exception as e:  # pylint: disable=W0703
            self.exception_occurred.emit(e)
            return
        if segment_data is None:
            return
        self.execution_done.emit(segment_data)

    def finished_task(self):
        """
        Called on calculation finished. Check if cache is not empty.
        In such case start calculation again with new parameters.
        """
        self.mutex.lock()
        if self.cache is not None:
            args, kwargs = self.cache
            self.algorithm.set_parameters(*args, **kwargs)
            self.cache = None
            self.clean_later = False
        if self._image is not None:
            self.algorithm.set_image(self._image)
            self._image = None
        if self._mask is not None:
            self.algorithm.set_mask(self._mask)
            self._mask = None
        if self.rerun[0]:
            self.rerun = False, QThread.InheritPriority
            super().start(self.rerun[1])
        elif self.clean_later:
            self.algorithm.clean()
            self.clean_later = False
        self.mutex.unlock()

    def clean(self):
        """
        clean cache if thread is running. Call :py:meth:`SegmentationAlgorithm.clean` otherwise. :
        """
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = True
        else:
            self.algorithm.clean()
        self.mutex.unlock()

    def set_parameters(self, *args, **kwargs):
        """
        check if calculation is running.
        If yes then cache parameters until it finish, otherwise call :py:meth:`.SegmentationAlgorithm.set_parameters`
        """
        self.mutex.lock()
        if self.isRunning():
            self.cache = args, kwargs
            self.clean_later = False
        else:
            self.algorithm.set_parameters(*args, **kwargs)
        self.mutex.unlock()

    def set_image(self, image):
        """
        check if calculation is running.
        If yes then cache parameters until it finish, otherwise call :py:meth:`.SegmentationAlgorithm.set_image`
        :param image: image to be set
        """
        self.mutex.lock()
        if self.isRunning():
            self._image = image
        else:
            self.algorithm.set_image(image)
        self.mutex.unlock()

    def set_mask(self, mask):
        """
        check if calculation is running.
        If yes then cache parameters until it finish, otherwise call :py:meth:`.SegmentationAlgorithm.set_mask`
        :param mask: mask to be set
        """
        self.mutex.lock()
        if self.isRunning():
            self._mask = mask
        else:
            self.algorithm.set_mask(mask)
        self.mutex.unlock()

    def start(self, priority: "QThread.Priority" = QThread.InheritPriority):
        """
        If calculation is running remember to restart it with new parameters.

        Otherwise start immediately.
        """
        self.mutex.lock()
        if self.isRunning():
            self.clean_later = False
            self.rerun = True, priority
        else:
            super().start(priority)
        self.mutex.unlock()
