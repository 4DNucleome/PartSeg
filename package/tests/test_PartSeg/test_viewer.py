import numpy as np
from qtpy.QtCore import QCoreApplication

from PartSeg._roi_analysis.image_view import ResultImageView
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSegCore.roi_info import ROIInfo


class TestResultImageView:
    def test_simple(self, qtbot, part_settings, image):
        prop = ChannelProperty(part_settings, "test")
        viewer = ResultImageView(part_settings, prop, "test")
        viewer.show()
        qtbot.add_widget(prop)
        qtbot.add_widget(viewer)
        viewer.add_image(image)
        assert not viewer.roi_alternative_select.isVisible()
        assert not viewer.label1.isVisible()
        assert not viewer.label2.isVisible()
        assert not viewer.opacity.isVisible()
        assert not viewer.only_border.isVisible()
        assert not viewer.roi_alternative_select.isVisible()
        assert not viewer.any_roi()
        assert not viewer.available_alternatives()
        viewer.hide()

    def test_set_roi(self, qtbot, part_settings, image):
        prop = ChannelProperty(part_settings, "test")
        viewer = ResultImageView(part_settings, prop, "test")
        qtbot.add_widget(prop)
        qtbot.add_widget(viewer)
        viewer.show()
        part_settings.image = image
        roi = ROIInfo((image.get_channel(0) > 0).astype(np.uint8))
        roi = roi.fit_to_image(image)
        viewer.set_roi(roi, image)
        QCoreApplication.processEvents()
        assert not viewer.roi_alternative_select.isVisible()
        assert viewer.label1.isVisible()
        assert viewer.label2.isVisible()
        assert viewer.opacity.isVisible()
        assert viewer.only_border.isVisible()
        assert not viewer.roi_alternative_select.isVisible()
        assert viewer.any_roi()
        assert not viewer.available_alternatives()
        viewer.hide()
