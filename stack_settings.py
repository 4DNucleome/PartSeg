from qt_import import QObject, pyqtSignal
import numpy as np


class ImageSettings(QObject):
    """
    :type _image: np.ndarray
    """
    image_changed = pyqtSignal([np.ndarray], [int])

    def __init__(self):
        super(ImageSettings, self).__init__()
        self.open_directory = ""
        self._image = None
        
    @property
    def image(self):
        return self._image
    
    @image.setter
    def image(self, value):
        value = np.squeeze(value)
        self._image = value
        if len(value.shape) == 4:
            if value.shape[-1] > 10:
                self._image = np.swapaxes(value, 1, 3)
                self._image = np.swapaxes(self._image, 1, 2)
        self.image_changed.emit(self._image)
        self.image_changed[int].emit(self.channels)

    @property
    def channels(self):
        if self._image is None:
            return 0
        if len(self._image.shape) == 4:
            return self._image.shape[-1]
        else:
            return 1
