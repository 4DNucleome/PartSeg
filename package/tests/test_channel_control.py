import PartSegData
import numpy as np
from qtpy.QtCore import Qt, QPoint
from qtpy.QtGui import QImage

from PartSeg.common_gui.channel_control import ColorComboBox, ColorComboBoxGroup, ChannelProperty
from PartSeg.common_gui.stack_image_view import ImageView
from PartSeg.project_utils_qt.settings import ViewSettings
from PartSeg.tiff_image import ImageReader
from PartSeg.utils.color_image import color_image


def array_from_image(image: QImage):
    size = image.size().width() * image.size().height()
    return np.frombuffer(image.bits().asstring(size * 3), dtype=np.uint8)


def test_color_combo_box(qtbot):
    box = ColorComboBox(0, ["BlackBlue", "BlackGreen", "BlackMagenta", "BlackRed", "gray"])
    box.show()
    qtbot.add_widget(box)
    with qtbot.waitSignal(box.channel_visible_changed):
        with qtbot.assertNotEmitted(box.clicked):
            qtbot.mouseClick(box.check_box, Qt.LeftButton)
    with qtbot.waitSignal(box.clicked, timeout=1000):
        qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(5, 5))
    with qtbot.waitSignal(box.clicked):
        qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(box.width() - 5, 5))

    box.set_color("BlackMagenta")
    img = color_image(np.arange(0, 256).reshape((1, 256, 1)), ["BlackMagenta"], [(0, 256)])
    assert np.all(array_from_image(box.image) == img.flatten())


class TestColorComboBoxGroup:
    def test_change_channels_num(self, qtbot):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", height=30)
        qtbot.add_widget(box)
        box.set_channels(1)
        box.set_channels(4)
        box.set_channels(10)
        box.set_channels(4)
        box.set_channels(10)
        box.set_channels(2)

    def test_color_combo_box_group(self, qtbot):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", height=30)
        qtbot.add_widget(box)
        box.set_channels(3)
        assert len(box.current_colors) == 3
        assert all(map(lambda x: isinstance(x, str), box.current_colors))
        with qtbot.waitSignal(box.coloring_update):
            box.layout().itemAt(0).widget().check_box.setChecked(False)
        with qtbot.waitSignal(box.coloring_update):
            box.layout().itemAt(0).widget().setCurrentIndex(2)
        assert box.current_colors[0] is None
        assert all(map(lambda x: isinstance(x, str), box.current_colors[1:]))

    def test_color_combo_box_group_and_color_preview(self, qtbot):
        settings = ViewSettings()
        ch_property = ChannelProperty(settings, "test")
        box = ColorComboBoxGroup(settings, "test", ch_property, height=30)
        qtbot.add_widget(box)
        qtbot.add_widget(ch_property)
        box.set_channels(3)
        box.set_active(1)
        with qtbot.assert_not_emitted(box.coloring_update), qtbot.assert_not_emitted(box.change_channel):
            ch_property.minimum_value.setValue(10)
            ch_property.minimum_value.setValue(100)

        def check_parameters(name, index):
            return name == "test" and index == 1

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.fixed.setChecked(True)

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.minimum_value.setValue(10)

        ch_property.maximum_value.setValue(10000)

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.maximum_value.setValue(11000)

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.fixed.setChecked(False)

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.use_gauss.setChecked(True)

        ch_property.gauss_radius.setValue(0.5)
        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.gauss_radius.setValue(1)

        with qtbot.waitSignal(box.coloring_update), \
                qtbot.waitSignal(box.change_channel, check_params_cb=check_parameters):
            ch_property.use_gauss.setChecked(False)

        with qtbot.assert_not_emitted(box.coloring_update), qtbot.assert_not_emitted(box.change_channel):
            ch_property.gauss_radius.setValue(0.5)

    def test_image_view_integration(self, qtbot):
        settings = ViewSettings()
        ch_property = ChannelProperty(settings, "test")
        image_view = ImageView(settings, ch_property, "test")
        qtbot.addWidget(image_view)
        qtbot.addWidget(ch_property)
        image = ImageReader.read_image(PartSegData.segmentation_analysis_default_image)
        with qtbot.waitSignal(settings.image_changed):
            settings.image = image
        channels_num = image.channels
        assert image_view.channel_control.channels_count == channels_num

        assert image_view.stack_slider.isVisible() is False
        assert image_view.time_slider.isVisible() is False
