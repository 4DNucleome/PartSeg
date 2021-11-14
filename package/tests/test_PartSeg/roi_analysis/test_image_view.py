from PartSeg._roi_analysis.image_view import CompareImageView, ResultImageView, SynchronizeView
from PartSeg.common_gui.channel_control import ChannelProperty


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
    view1.viewer.dims.set_point(len(point1) - 1, point1[-1])
    assert view2.viewer.dims.point == point1
    assert view1.viewer.dims.point == point1
    view1.hide()
    view2.hide()
