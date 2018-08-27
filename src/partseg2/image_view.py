import collections

from PyQt5.QtGui import QHideEvent, QShowEvent
from scipy.ndimage import gaussian_filter

from common_gui.stack_image_view import ImageView
from project_utils.color_image import color_image
import numpy as np

class RawImageView(ImageView):
    pass

    def hideEvent(self, a0: QHideEvent):
        self.parent().layout().setColumnStretch(0, 0)

    def showEvent(self, event: QShowEvent):
        self.parent().layout().setColumnStretch(0, 1)

    def add_labels(self, im):
        return im

    def info_text_pos(self, *pos):
        if self.tmp_image is None:
            return

        brightness = self.tmp_image[pos if len(pos) == self.tmp_image.ndim -1 else pos[1:]]
        pos2 = list(pos)
        pos2[0] += 1
        if isinstance(brightness, collections.Iterable):
            res_brightness = []
            for i, b in enumerate(brightness):
                if self.channel_control.active_channel(i):
                    res_brightness.append(b)
            brightness = res_brightness
            if len(brightness) == 1:
                brightness = brightness[0]

        self.text_info_change.emit("Position: {}, Brightness: {}".format(tuple(pos2), brightness))

class ResultImageView(ImageView):
    pass
