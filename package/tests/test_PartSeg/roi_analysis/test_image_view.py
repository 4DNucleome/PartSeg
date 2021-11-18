# pylint: disable=R0201
import platform

import numpy as np
import pytest

from PartSeg._roi_analysis.image_view import CompareImageView, ResultImageView, SynchronizeView
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSegCore.roi_info import ROIInfo

from ..utils import CI_BUILD


@pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
def test_synchronize(part_settings, image2, qtbot):
    prop = ChannelProperty(part_settings, "test1")
    view1 = ResultImageView(part_settings, prop, "test1")
    view2 = CompareImageView(part_settings, prop, "test2")
    sync = SynchronizeView(view1, view2)
    qtbot.add_widget(prop)
    qtbot.add_widget(view1)
    qtbot.add_widget(view2)
    view1.show()
    view2.show()
    part_settings.image = image2
    point1 = view1.viewer.dims.point
    point2 = view2.viewer.dims.point
    sync.set_synchronize(False)
    view1.viewer.dims.set_point(len(point1) - 1, point1[-1] + 1 * view1.viewer.dims.range[-1][-1])
    assert view1.viewer.dims.point != point1
    assert view2.viewer.dims.point == point2
    sync.set_synchronize(True)
    view1.viewer.dims.set_point(len(point1) - 1, point1[-1] + 2 * view1.viewer.dims.range[-1][-1])
    assert view2.viewer.dims.point != point2
    assert view2.viewer.dims.point == view1.viewer.dims.point
    view2.viewer.dims.set_point(len(point1) - 1, point1[-1])
    assert view2.viewer.dims.point == point1
    assert view1.viewer.dims.point == point1
    view1.hide()
    view2.hide()


def test_compare_view(part_settings, image2, qtbot):
    prop = ChannelProperty(part_settings, "test")
    view = CompareImageView(part_settings, prop, "test")
    qtbot.add_widget(prop)
    qtbot.add_widget(view)
    part_settings.image = image2

    roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)

    assert view.image_info[str(image2.file_path)].roi is None
    part_settings.roi = roi
    assert view.image_info[str(image2.file_path)].roi is None
    part_settings.set_segmentation_to_compare(ROIInfo(roi))
    assert view.image_info[str(image2.file_path)].roi is not None


class TestResultImageView:
    def test_setting_properties(self, part_settings, image2, qtbot):
        prop = ChannelProperty(part_settings, "test")
        view = ResultImageView(part_settings, prop, "test")
        qtbot.add_widget(prop)
        qtbot.add_widget(view)
        part_settings.image = image2

        roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)

        assert view.image_info[str(image2.file_path)].roi is None
        with qtbot.waitSignal(part_settings.roi_changed):
            part_settings.roi = roi
        assert view.any_roi()
        assert view.image_info[str(image2.file_path)].roi is not None
        assert part_settings.get_from_profile("test.image_state.opacity") == 1
        view.opacity.setValue(0.5)
        assert part_settings.get_from_profile("test.image_state.opacity") == 0.5
        assert view.image_info[str(image2.file_path)].roi.opacity == 0.5
        part_settings.set_in_profile("test.image_state.opacity", 1)
        assert view.opacity.value() == 1
        assert view.image_info[str(image2.file_path)].roi.opacity == 1

        assert not part_settings.get_from_profile("test.image_state.only_border")
        view.only_border.setChecked(True)
        assert part_settings.get_from_profile("test.image_state.only_border")
        part_settings.set_in_profile("test.image_state.only_border", False)
        assert not view.only_border.isChecked()

    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_resize(self, part_settings, image2, qtbot):
        prop = ChannelProperty(part_settings, "test")
        view = ResultImageView(part_settings, prop, "test")
        qtbot.add_widget(prop)
        qtbot.add_widget(view)
        part_settings.image = image2

        view.show()
        view.resize(500, 500)
        assert view.btn_layout2.count() == 2
        view.resize(1000, 500)
        assert view.btn_layout2.count() == 0
        view.resize(500, 500)
        assert view.btn_layout2.count() == 2
        view.hide()

    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_with_roi_alternatives(self, part_settings, image2, qtbot):
        prop = ChannelProperty(part_settings, "test")
        view = ResultImageView(part_settings, prop, "test")
        qtbot.add_widget(prop)
        qtbot.add_widget(view)
        part_settings.image = image2
        view.show()

        roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        part_settings.roi = roi

        assert not view.roi_alternative_select.isVisible()
        part_settings.roi = ROIInfo(roi, alternative={"t1": roi, "t2": roi})
        assert view.roi_alternative_select.isVisible()
        assert view.roi_alternative_select.count() == 3

        assert view.roi_alternative_select.currentText() == "ROI"
        view.roi_alternative_select.setCurrentText("t1")

        view.hide()
