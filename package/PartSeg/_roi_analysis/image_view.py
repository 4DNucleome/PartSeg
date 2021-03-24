from typing import Optional

from qtpy.QtCore import QObject, QSignalBlocker, Slot
from qtpy.QtGui import QResizeEvent
from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QLabel

from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image

from ..common_gui.channel_control import ChannelProperty
from ..common_gui.napari_image_view import ImageView
from .partseg_settings import PartSettings


class ResultImageView(ImageView):
    """
    :type _settings PartSettings:
    """

    def __init__(self, settings: PartSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        self._channel_control_top = True
        self.only_border = QCheckBox("")
        self.image_state.only_borders = False
        self.only_border.setChecked(self.image_state.only_borders)
        self.only_border.stateChanged.connect(self.image_state.set_borders)
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 1)
        self.opacity.setValue(self.image_state.opacity)
        self.opacity.setSingleStep(0.1)
        self.opacity.valueChanged.connect(self.image_state.set_opacity)
        self.opacity.setMinimumWidth(500)
        self.channel_control_index = self.btn_layout.indexOf(self.channel_control)
        self.label1 = QLabel("Borders:")
        self.label2 = QLabel("Opacity:")
        self.roi_alternative_select = QComboBox()
        self.roi_alternative_select.currentTextChanged.connect(self.image_state.set_roi_presented)

        self.btn_layout.insertWidget(self.channel_control_index + 1, self.label1)
        self.btn_layout.insertWidget(self.channel_control_index + 2, self.only_border)
        self.btn_layout.insertWidget(self.channel_control_index + 3, self.label2)
        self.btn_layout.insertWidget(self.channel_control_index + 4, self.opacity)
        self.btn_layout.insertWidget(self.channel_control_index + 1, self.roi_alternative_select)
        self.label1.setVisible(False)
        self.label2.setVisible(False)
        self.opacity.setVisible(False)
        self.only_border.setVisible(False)
        self.roi_alternative_select.setVisible(False)

    def any_roi(self):
        return any(image_info.roi is not None for image_info in self.image_info.values())

    def available_alternatives(self):
        available_alternatives = set()
        for image_info in self.image_info.values():
            if image_info.roi_info.alternative:
                available_alternatives.update(image_info.roi_info.alternative.keys())
        return available_alternatives

    @Slot()
    @Slot(ROIInfo)
    def set_roi(self, roi_info: Optional[ROIInfo] = None, image: Optional[Image] = None) -> None:
        super().set_roi(roi_info, image)
        show = self.any_roi()
        self.label1.setVisible(show)
        self.label2.setVisible(show)
        self.opacity.setVisible(show)
        self.only_border.setVisible(show)
        self.update_alternatives()

    def update_alternatives(self):
        alternatives = self.available_alternatives()
        self.roi_alternative_select.setVisible(bool(alternatives))
        text = self.roi_alternative_select.currentText()
        block = self.roi_alternative_select.signalsBlocked()
        self.roi_alternative_select.blockSignals(True)
        self.roi_alternative_select.clear()
        self.roi_alternative_select.addItems(["ROI"] + list(alternatives))
        self.roi_alternative_select.setCurrentText(text)
        self.roi_alternative_select.blockSignals(block)

    def resizeEvent(self, event: QResizeEvent):
        if event.size().width() > 700 and not self._channel_control_top:
            w = self.btn_layout2.takeAt(0).widget()
            self.btn_layout.takeAt(self.channel_control_index)
            # noinspection PyArgumentList
            self.btn_layout.insertWidget(self.channel_control_index, w)
            select = self.btn_layout2.takeAt(self.btn_layout2.indexOf(self.roi_alternative_select)).widget()
            self.btn_layout.insertWidget(self.channel_control_index + 1, select)
            self._channel_control_top = True
        elif event.size().width() <= 700 and self._channel_control_top:
            w = self.btn_layout.takeAt(self.channel_control_index).widget()
            self.btn_layout.insertStretch(self.channel_control_index, 1)
            # noinspection PyArgumentList
            self.btn_layout2.insertWidget(0, w)
            select = self.btn_layout.takeAt(self.btn_layout.indexOf(self.roi_alternative_select)).widget()
            self.btn_layout2.insertWidget(1, select)
            self._channel_control_top = False


class CompareImageView(ResultImageView):
    def __init__(self, settings: PartSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        settings.roi_changed.disconnect(self.set_roi)
        settings.roi_clean.disconnect(self.set_roi)
        settings.compare_segmentation_change.connect(self.set_roi)


class SynchronizeView(QObject):
    def __init__(self, image_view1: ImageView, image_view2: ImageView, parent=None):
        super().__init__(parent)
        self.image_view1 = image_view1
        self.image_view2 = image_view2
        self.synchronize = False
        self.image_view1.view_changed.connect(self.synchronize_views)
        self.image_view2.view_changed.connect(self.synchronize_views)

    def set_synchronize(self, val: bool):
        self.synchronize = val

    @Slot()
    def synchronize_views(self):
        if not self.synchronize or self.image_view1.isHidden() or self.image_view2.isHidden():
            return
        sender = self.sender()
        if sender == self.image_view1:
            origin, dest = self.image_view1, self.image_view2
        else:
            origin, dest = self.image_view2, self.image_view1
        _block = QSignalBlocker(dest)  # noqa F841
        dest.set_state(origin.get_state())
