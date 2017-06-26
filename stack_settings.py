from qt_import import QObject, pyqtSignal
import numpy as np


class Settings(QObject):
    image_changed = pyqtSignal(np.ndarray)

    def __init__(self):
        super(Settings, self).__init__()
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
