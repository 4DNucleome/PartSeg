import collections

from PyQt5.QtGui import QHideEvent, QShowEvent
from PyQt5.QtWidgets import QPushButton, QStackedWidget
from scipy.ndimage import gaussian_filter

from common_gui.channel_control import ChannelControl
from common_gui.stack_image_view import ImageView, create_tool_button
from partseg2.advanced_window import StatisticsWindow
from project_utils.color_image import color_image
import numpy as np

class RawImageStack(QStackedWidget):
    def __init__(self, settings, channel_control:ChannelControl):
        super().__init__()
        self.raw_image = RawImageView(settings, channel_control)
        self.statistic_calculate = StatisticsWindowForRaw(settings)
        self.addWidget(self.raw_image)
        self.addWidget(self.statistic_calculate)

class StatisticsWindowForRaw(StatisticsWindow):
    def __init__(self, settings):
        super().__init__(settings)
        self.image_view = QPushButton("Image preview")
        self.up_butt_layout.addWidget(self.image_view)
        self.image_view.clicked.connect(self.image_view_fun)

    def image_view_fun(self):
        self.parent().setCurrentIndex(0)

class RawImageView(ImageView):
    def __init__(self, settings, channel_control: ChannelControl):
        super().__init__(settings, channel_control)
        self.statistic_image_view_btn = create_tool_button("Statistic calculation", None)
        self.btn_layout.addWidget(self.statistic_image_view_btn)
        self.statistic_image_view_btn.clicked.connect(self.show_statistic)

    def show_statistic(self):
        self.parent().setCurrentIndex(1)

    def hideEvent(self, a0: QHideEvent):
        self.parent().parent().layout().setColumnStretch(0, 0)

    def showEvent(self, event: QShowEvent):
        self.parent().parent().layout().setColumnStretch(0, 1)

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
