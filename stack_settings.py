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
        return 
    
    @image.setter
    def image(self, value):
        pass
