import collections

from qtpy import QtCore
from qtpy.QtCore import QObject, QEvent
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel

from ..common_gui.channel_control import ChannelProperty
from ..common_gui.stack_image_view import ImageView
from .partseg_settings import MASK_COLORS, PartSettings
import numpy as np


class ImageViewWithMask(ImageView):
    """
    :type _settings PartSettings:
    """
    def __init__(self, settings: PartSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self.mask_show = QCheckBox()
        self.mask_label = QLabel("Mask:")
        # self.btn_layout.takeAt(self.btn_layout.count()-1)
        self.btn_layout.addWidget(self.mask_label)
        self.btn_layout.addWidget(self.mask_show)
        self.mask_prop = self._settings.get_from_profile("mask_presentation", (list(MASK_COLORS.keys())[0], 1))
        self.mask_show.setDisabled(True)
        self.mask_label.setDisabled(True)
        settings.mask_changed.connect(self.mask_changed)
        self.mask_show.stateChanged.connect(self.paint_layer)

    def event(self, event: QtCore.QEvent):
        if event.type() == QEvent.WindowActivate:
            if self.mask_show.isChecked():
                color, opacity = self._settings.get_from_profile("mask_presentation")
                if color != self.mask_prop[0] or opacity != self.mask_prop[1]:
                    self.mask_prop = color, opacity
                    self.paint_layer()
        return super().event(event)

    def mask_changed(self):
        self.mask_show.setDisabled(self._settings.mask is None)
        self.mask_label.setDisabled(self._settings.mask is None)
        if self._settings.mask is None:
            self.mask_show.setChecked(False)
        elif self.mask_show.isChecked():
            self.paint_layer()

    def add_mask(self, im):
        if not self.mask_show.isChecked() or self._settings.mask is None:
            return
        mask_layer = self._settings.mask[self.stack_slider.value()]
        mask_layer = mask_layer.astype(np.bool)

        if self.mask_prop[1] == 1:
            im[~mask_layer] = MASK_COLORS[self.mask_prop[0]]
        else:
            im[~mask_layer] = (1 - self.mask_prop[1]) * im[~mask_layer] + \
                              self.mask_prop[1] * MASK_COLORS[self.mask_prop[0]]

    def set_image(self):
        super().set_image()
        self.mask_changed()


class RawImageView(ImageViewWithMask):
    def add_labels(self, im):
        return im


class ResultImageView(ImageViewWithMask):
    def __init__(self, settings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self.only_border = QCheckBox("")
        self.image_state.only_borders = False
        self.only_border.setChecked(self.image_state.only_borders)
        self.only_border.stateChanged.connect(self.image_state.set_borders)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setValue(self.image_state.opacity)
        self.opacity.setSingleStep(0.1)
        self.opacity.valueChanged.connect(self.image_state.set_opacity)

        self.btn_layout.addWidget(QLabel("Borders:"))
        self.btn_layout.addWidget(self.only_border)
        self.btn_layout.addWidget(QLabel("Opacity:"))
        self.btn_layout.addWidget(self.opacity)


class SynchronizeView(QObject):
    def __init__(self, image_view1: ImageView, image_view2: ImageView, parent=None):
        super(). __init__(parent)
        self.image_view1 = image_view1
        self.image_view2 = image_view2
        self.synchronize = False
        self.image_view1.stack_slider.sliderMoved.connect(self.synchronize_sliders)
        self.image_view2.stack_slider.sliderMoved.connect(self.synchronize_sliders)
        self.image_view1.image_area.zoom_changed.connect(self.synchronize_zoom)
        self.image_view2.image_area.zoom_changed.connect(self.synchronize_zoom)
        self.image_view1.image_area.horizontalScrollBar().valueChanged.connect(self.synchronize_shift)
        self.image_view2.image_area.horizontalScrollBar().valueChanged.connect(self.synchronize_shift)
        self.image_view1.image_area.verticalScrollBar().valueChanged.connect(self.synchronize_shift)
        self.image_view2.image_area.verticalScrollBar().valueChanged.connect(self.synchronize_shift)

    def set_synchronize(self, val: bool):
        self.synchronize = val

    def synchronize_sliders(self, val):
        if not self.synchronize or self.image_view1.isHidden() or self.image_view2.isHidden():
            return
        if self.sender() == self.image_view2.stack_slider:
            self.image_view1.stack_slider.setValue(val)
        else:
            self.image_view2.stack_slider.setValue(val)

    def synchronize_zoom(self):
        if not self.synchronize or self.image_view1.isHidden() or self.image_view2.isHidden():
            return
        if self.sender() == self.image_view1.image_area:
            origin = self.image_view1.image_area
            dest = self.image_view2.image_area
        else:
            origin = self.image_view2.image_area
            dest = self.image_view1.image_area

        dest.zoom_scale = origin.zoom_scale
        dest.x_mid = origin.x_mid
        dest.y_mid = origin.y_mid
        dest.resize_pixmap()

    def synchronize_shift(self):
        if not self.synchronize or self.image_view1.isHidden() or self.image_view2.isHidden():
            return
        if self.sender().parent() == self.image_view1.image_area:
            origin = self.image_view1.image_area
            dest = self.image_view2.image_area
        else:
            origin = self.image_view2.image_area
            dest = self.image_view1.image_area
        dest.horizontalScrollBar().setValue(origin.horizontalScrollBar().value())
        dest.verticalScrollBar().setValue(origin.verticalScrollBar().value())
