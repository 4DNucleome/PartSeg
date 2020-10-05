import numpy as np
from qtpy.QtCore import QThread
from scipy.ndimage import zoom


class InterpolateThread(QThread):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.scaling = None
        self.arrays = None
        self.result = None

    def set_scaling(self, scaling):
        self.scaling = scaling

    def set_arrays(self, arrays_list):
        self.arrays = arrays_list

    def run(self):
        self.result = []
        for el in self.arrays:
            if len(el.shape) == len(self.scaling):
                cache = zoom(el, self.scaling, mode="mirror")

            else:
                shape = [round(x * y) for x, y in zip(self.scaling, el.shape)] + list(el.shape[len(self.scaling) :])
                cache = np.zeros(shape, dtype=el.dtype)
                for i in range(el.shape[-1]):
                    cache[..., i] = zoom(el[..., i], self.scaling, mode="mirror")
            self.result.append(cache)
