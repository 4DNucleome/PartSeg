from qt_import import QObject, pyqtSignal
import numpy as np


class ImageSettings(QObject):
    """
    :type _image: np.ndarray
    """
    image_changed = pyqtSignal([np.ndarray], [int], [str])

    def __init__(self):
        super(ImageSettings, self).__init__()
        self.open_directory = ""
        self._image = None
        self._image_path = ""
        self.has_channels = False
        self.image_spacing = 70, 70, 210
        self.segmentation = None

    @property
    def batch_directory(self):
        return self.open_directory

    @batch_directory.setter
    def batch_directory(self, val):
        self.open_directory = val
        
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, value):
        if isinstance(value, tuple):
            file_path = value[1]
            value = value[0]
        else:
            file_path = None
        value = np.squeeze(value)
        self._image = value
        if len(value.shape) == 4:
            if value.shape[-1] > 10:
                self._image = np.swapaxes(value, 1, 3)
                self._image = np.swapaxes(self._image, 1, 2)
        if file_path is not None:
            self._image_path = file_path
            self.image_changed[str].emit(self._image_path)
        if self._image.shape[-1] < 10:
            self.has_channels = True
        else:
            self.has_channels = False
        self.image_changed.emit(self._image)
        self.image_changed[int].emit(self.channels)

    @property
    def image_path(self):
        return self._image_path

    @image_path.setter
    def image_path(self, value):
        self._image_path = value
        self.image_changed[str].emmit(self._image_path)

    @property
    def channels(self):
        if self._image is None:
            return 0
        if len(self._image.shape) == 4:
            return self._image.shape[-1]
        else:
            return 1

    def get_chanel(self, chanel_num):
        if self.has_channels:
            return self._image[..., chanel_num]
        return self._image
    
    def get_information(self, *pos):
        return self._image[pos]

