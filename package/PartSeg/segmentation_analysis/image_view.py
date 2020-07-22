from typing import Optional

from qtpy.QtCore import QObject, QSignalBlocker, Slot
from qtpy.QtGui import QResizeEvent
from qtpy.QtWidgets import QCheckBox, QDoubleSpinBox, QLabel

from PartSegCore.segmentation_info import SegmentationInfo
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
        self.label1 = QLabel("Borders:")
        self.label2 = QLabel("Opacity:")
        self.btn_layout.insertWidget(3, self.label1)
        self.btn_layout.insertWidget(4, self.only_border)
        self.btn_layout.insertWidget(5, self.label2)
        self.btn_layout.insertWidget(6, self.opacity)
        self.label1.setVisible(False)
        self.label2.setVisible(False)
        self.opacity.setVisible(False)
        self.only_border.setVisible(False)

    def any_segmentation(self):
        for image_info in self.image_info.values():
            if image_info.segmentation is not None:
                return True
        return False

    @Slot()
    @Slot(SegmentationInfo)
    def set_segmentation(
        self, segmentation_info: Optional[SegmentationInfo] = None, image: Optional[Image] = None
    ) -> None:
        super().set_segmentation(segmentation_info, image)
        show = self.any_segmentation()
        self.label1.setVisible(show)
        self.label2.setVisible(show)
        self.opacity.setVisible(show)
        self.only_border.setVisible(show)

    def resizeEvent(self, event: QResizeEvent):
        if event.size().width() > 700 and not self._channel_control_top:
            w = self.btn_layout2.takeAt(0).widget()
            self.btn_layout.takeAt(2)
            # noinspection PyArgumentList
            self.btn_layout.insertWidget(2, w)
            self._channel_control_top = True
        elif event.size().width() <= 700 and self._channel_control_top:
            w = self.btn_layout.takeAt(2).widget()
            self.btn_layout.insertStretch(2, 1)
            # noinspection PyArgumentList
            self.btn_layout2.insertWidget(0, w)
            self._channel_control_top = False


class CompareImageView(ResultImageView):
    def __init__(self, settings: PartSettings, channel_property: ChannelProperty, name: str):
        super().__init__(settings, channel_property, name)
        settings.segmentation_changed.disconnect(self.set_segmentation)
        settings.segmentation_clean.disconnect(self.set_segmentation)
        settings.compare_segmentation_change.connect(self.set_segmentation)


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
        # print("AAAAA")
        if not self.synchronize or self.image_view1.isHidden() or self.image_view2.isHidden():
            return
        sender = self.sender()
        if sender == self.image_view1:
            origin, dest = self.image_view1, self.image_view2
        else:
            origin, dest = self.image_view2, self.image_view1
        _block = QSignalBlocker(dest)  # noqa F841
        dest.set_state(origin.get_state())
