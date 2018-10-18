from tiff_image import ImageReader, Image
from .progress_thread import ProgressTread
from PyQt5.QtCore import pyqtSignal


class ImageReaderThread(ProgressTread):
    """
    thread for reading files. Useful for reading from disc
    """
    image_read_finish = pyqtSignal(Image)

    def __init__(self, file_path=None, mask_path=None, parent=None):
        super().__init__(parent)
        self.reader = ImageReader(self.info_function)
        self.file_path = file_path
        self.mask_path = mask_path
        self.image = None

    def set_path(self, file_path, mask_path=None):
        self.file_path = file_path
        self.mask_path = mask_path

    def run(self):
        if self.file_path is None:
            return
        try:
            self.image = self.reader.read(self.file_path, self.mask_path)
            self.image_read_finish.emit(self.image)
        except Exception as e:
            self.error_signal.emit(e)

    def info_function(self, label: str, val: int):
        if label == "max":
            self.range_changed.emit(0, val)
        elif label == "step":
            self.step_changed.emit(val)
