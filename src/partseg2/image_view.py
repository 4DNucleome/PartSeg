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

    def change_image(self):
        if self.image is None:
            return
        img = np.copy(self.image[self.stack_slider.value()])
        color_maps = self.channel_control.current_colors
        borders = self.border_val[:]
        for i, p in enumerate(self.channel_control.get_limits()):
            if p is not None:
                borders[i] = p
        for i, (use, radius) in enumerate(self.channel_control.get_gauss()):
            if use and color_maps[i] is not None and radius > 0:
                img[..., i] = gaussian_filter(img[..., i], radius)
        im = color_image(img, color_maps, borders)
        self.image_area.set_image(im, self.sender() is not None)
        self.tmp_image = np.array([img])

    def info_text_pos(self, *pos):
        if self.tmp_image is None:
            return
        brightness = self.tmp_image[pos]
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
