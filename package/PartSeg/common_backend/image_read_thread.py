from qtpy.QtCore import Signal

from PartSegImage import Image, TiffImageReader

from .progress_thread import ProgressTread


class ImageReaderThread(ProgressTread):
    """
    thread for reading files. Useful for reading from disc
    """

    image_read_finish = Signal(Image)

    def __init__(self, file_path=None, mask_path=None):
        super().__init__()
        self.reader = TiffImageReader(self.info_function)
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
