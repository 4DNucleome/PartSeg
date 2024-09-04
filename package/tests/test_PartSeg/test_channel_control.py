# pylint: disable=no-self-use

from unittest.mock import MagicMock

import numpy as np
import pytest
from napari.utils.colormaps import make_colorbar
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QImage

import PartSegData
from PartSeg.common_backend.base_settings import BaseSettings, ColormapDict, ViewSettings
from PartSeg.common_gui.channel_control import ChannelProperty, ColorComboBox, ColorComboBoxGroup
from PartSeg.common_gui.napari_image_view import ImageView
from PartSegCore.color_image.base_colors import starting_colors
from PartSegCore.image_operations import NoiseFilterType
from PartSegImage import TiffImageReader

try:
    from qtpy import PYQT5, PYQT6
except ImportError:  # pragma: no cover
    PYQT5 = True
    PYQT6 = False

if PYQT5 or PYQT6:

    def array_from_image(image: QImage):
        size = image.size().width() * image.size().height()
        return np.frombuffer(image.bits().asstring(size * image.depth() // 8), dtype=np.uint8)

else:

    def array_from_image(image: QImage):
        size = image.size().width() * image.size().height()
        return np.frombuffer(image.bits(), dtype=np.uint8, count=size * image.depth() // 8)


@pytest.fixture
def base_settings(tmp_path, qapp):
    return BaseSettings(tmp_path)


@pytest.fixture
def ch_property(base_settings, qtbot):
    ch_prop = ChannelProperty(base_settings, start_name="test")
    qtbot.add_widget(ch_prop)
    return ch_prop


@pytest.fixture
def image_view(base_settings, ch_property, qtbot):
    image_view = ImageView(base_settings, ch_property, "test")
    qtbot.add_widget(image_view)
    image = TiffImageReader.read_image(PartSegData.segmentation_analysis_default_image)
    with qtbot.waitSignal(image_view.image_added, timeout=10**6):
        base_settings.image = image

    channels_num = image.channels
    assert image_view.channel_control.channels_count == channels_num

    image_view.channel_control.set_active(1)

    return image_view


class TestChannelProperty:
    def test_fail_construct(self, base_settings):
        with pytest.raises(ValueError, match="non empty start_name"):
            ChannelProperty(base_settings, start_name="")

    def test_collapse(self, base_settings, qtbot):
        ch_prop = ChannelProperty(base_settings, start_name="test")
        qtbot.add_widget(ch_prop)
        ch_prop.show()
        assert not ch_prop.collapse_widget.isChecked()
        assert ch_prop.minimum_value.isVisible()
        ch_prop.collapse_widget.setChecked(True)
        assert not ch_prop.minimum_value.isVisible()
        ch_prop.hide()

    def test_get_value_from_settings(self, base_settings, qtbot):
        ch_prop = ChannelProperty(base_settings, start_name="test1")
        base_settings.set_in_profile("test.range_0", (100, 300))
        mock = MagicMock()
        mock.viewer_name = "test"
        ch_prop.register_widget(mock)
        with pytest.raises(ValueError, match="name test already register"):
            ch_prop.register_widget(mock)
        assert ch_prop.minimum_value.value() == 100
        assert ch_prop.maximum_value.value() == 300
        base_settings.set_in_profile("test.range_0", (200, 500))
        assert ch_prop.minimum_value.value() == 200
        assert ch_prop.maximum_value.value() == 500
        base_settings.set_in_profile("test.range_1", (20, 50))
        assert ch_prop.minimum_value.value() == 200
        assert ch_prop.maximum_value.value() == 500
        with pytest.raises(ValueError, match="name test7 not in register"):
            ch_prop.change_current("test7", 1)


class TestColorComboBox:
    def test_base(self, qtbot):
        dkt = ColormapDict({})
        box = ColorComboBox(0, starting_colors, dkt)
        box.show()
        qtbot.add_widget(box)
        with qtbot.waitSignal(box.channel_visible_changed), qtbot.assertNotEmitted(box.clicked):
            qtbot.mouseClick(box.check_box, Qt.LeftButton)
        with qtbot.waitSignal(box.clicked, timeout=1000):
            qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(5, 5))
        with qtbot.waitSignal(box.clicked):
            qtbot.mouseClick(box, Qt.LeftButton, pos=QPoint(box.width() - 5, 5))
        index = 3
        with qtbot.waitSignal(box.currentTextChanged):
            box.set_color(starting_colors[index])
        img = np.array(make_colorbar(dkt[starting_colors[index]][0], size=(1, 512)))
        assert np.all(array_from_image(box.image) == img.flatten())
        box.hide()

    def test_visibility(self, qtbot):
        dkt = ColormapDict({})
        box = ColorComboBox(0, starting_colors, dkt, lock=True)
        qtbot.add_widget(box)
        box.show()
        qtbot.wait(100)
        assert box.lock.isVisible()
        box.hide()
        box = ColorComboBox(0, starting_colors, dkt, blur=NoiseFilterType.Gauss)
        qtbot.add_widget(box)
        box.show()
        qtbot.wait(100)
        assert box.blur.isVisible()
        box.hide()
        box = ColorComboBox(0, starting_colors, dkt, gamma=2)
        qtbot.add_widget(box)
        box.show()
        qtbot.wait(100)
        assert box.gamma.isVisible()
        box.hide()

    def test_show_frame_arrow(self, qtbot):
        dkt = ColormapDict({})
        box = ColorComboBox(0, starting_colors, dkt)
        qtbot.add_widget(box)
        box.show()
        box.show_arrow = True
        box.repaint()
        qtbot.wait(100)
        box.show_arrow = False
        box.show_frame = True
        box.repaint()
        qtbot.wait(100)
        box.hide()

    def test_change_colors(self, qtbot):
        dkt = ColormapDict({})
        box = ColorComboBox(0, starting_colors, dkt)
        qtbot.add_widget(box)
        box.change_colors(starting_colors[:-1])
        assert box.count() == len(starting_colors) - 1
        box.change_colors(starting_colors[1:])
        assert box.count() == len(starting_colors) - 1

    def test_item_delegate(self, qtbot):
        dkt = ColormapDict({})
        box = ColorComboBox(0, starting_colors, dkt)
        qtbot.add_widget(box)
        box.show()
        box.showPopup()
        qtbot.wait(100)
        box.hide()


class TestColorComboBoxGroup:
    def test_change_channels_num(self, qtbot, image2):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", height=30)
        qtbot.add_widget(box)
        box.set_channels(1)
        box.set_channels(4)
        box.set_channels(10)
        box.set_channels(4)
        box.set_channels(10)
        settings.image = image2
        box.update_channels()
        assert box.layout().count() == image2.channels

    def test_update_colormaps(self, qtbot, base_settings):
        box = ColorComboBoxGroup(base_settings, "test", height=30)
        qtbot.add_widget(box)
        box.set_channels(4)
        assert box.current_colormaps == [base_settings.colormap_dict[x][0] for x in starting_colors[:4]]
        box.update_color_list(starting_colors[1:2])
        assert box.current_colors == [starting_colors[1] for _ in range(4)]
        box.update_color_list()
        assert box.layout().itemAt(0).widget().count() == len(starting_colors)

    def test_settings_updated(self, qtbot, base_settings, monkeypatch):
        box = ColorComboBoxGroup(base_settings, "test", height=30)
        box.set_channels(4)
        mock = MagicMock()
        monkeypatch.setattr(box, "parameters_changed", mock)
        base_settings.set_in_profile("test.lock_0", True)
        mock.assert_called_once_with(0)
        dkt = dict(**base_settings.get_from_profile("test"))
        dkt["lock_0"] = False
        dkt["lock_1"] = True
        base_settings.set_in_profile("test", dkt)
        assert mock.call_count == 5
        mock.assert_called_with(3)

    def test_color_combo_box_group(self, qtbot):
        settings = ViewSettings()
        box = ColorComboBoxGroup(settings, "test", height=30)
        qtbot.add_widget(box)
        box.set_channels(3)
        assert len(box.current_colors) == 3
        assert all(isinstance(x, str) for x in box.current_colors)
        with qtbot.waitSignal(box.coloring_update):
            box.layout().itemAt(0).widget().check_box.setChecked(False)
        with qtbot.waitSignal(box.coloring_update):
            box.layout().itemAt(0).widget().setCurrentIndex(2)
        assert box.current_colors[0] is None
        assert all(isinstance(x, str) for x in box.current_colors[1:])

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

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(True)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.minimum_value.setValue(10)

        ch_property.maximum_value.setValue(10000)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.maximum_value.setValue(11000)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(False)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.use_filter.setCurrentEnum(NoiseFilterType.Gauss)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.use_filter.setCurrentEnum(NoiseFilterType.Median)

        ch_property.filter_radius.setValue(0.5)
        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.filter_radius.setValue(2)

        with qtbot.waitSignal(box.coloring_update), qtbot.waitSignal(
            box.change_channel, check_params_cb=check_parameters
        ):
            ch_property.use_filter.setCurrentEnum(NoiseFilterType.No)

        with qtbot.assert_not_emitted(box.coloring_update), qtbot.assert_not_emitted(box.change_channel):
            ch_property.filter_radius.setValue(0.5)

    @pytest.mark.windows_ci_skip
    @pytest.mark.parametrize("filter_value", NoiseFilterType.__members__.values())
    def test_image_view_integration_filter(self, qtbot, tmp_path, filter_value, ch_property, image_view):
        image_view.channel_control.set_active(1)

        def check_parameters(name, index):
            return name == "test" and index == 1

        if filter_value is NoiseFilterType.No:
            with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
                image_view.channel_control.change_channel, check_params_cb=check_parameters
            ):
                ch_property.use_filter.setCurrentEnum(NoiseFilterType.Gauss)
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.use_filter.setCurrentEnum(filter_value)
        image4 = image_view.viewer_widget.screenshot()
        assert (filter_value != NoiseFilterType.No and np.any(image4 != 255)) or (
            filter_value == NoiseFilterType.No and np.any(image4 == 255)
        )

    @pytest.mark.windows_ci_skip
    def test_image_view_integration(self, qtbot, tmp_path, ch_property, image_view):
        image_view.viewer_widget.screenshot(flash=False)
        image1 = image_view.viewer_widget._render()
        assert np.any(image1 != 255)
        ch_property.minimum_value.setValue(100)
        ch_property.maximum_value.setValue(10000)
        ch_property.filter_radius.setValue(0.5)
        image2 = image_view.viewer_widget._render()
        assert np.any(image2 != 255)

        assert np.all(image1 == image2)

        def check_parameters(name, index):
            return name == "test" and index == 1

        # Test fixed range
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(True)

        image1 = image_view.viewer_widget._render()
        assert np.any(image1 != 255)
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.minimum_value.setValue(20)
        image2 = image_view.viewer_widget._render()
        assert np.any(image2 != 255)
        assert np.any(image1 != image2)

        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.maximum_value.setValue(11000)
        image3 = image_view.viewer_widget.screenshot(flash=False)
        assert np.any(image3 != 255)
        assert np.any(image2 != image3)
        assert np.any(image1 != image3)

        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(False)

        image1 = image_view.viewer_widget.screenshot(flash=False)
        assert np.any(image1 != 255)
        assert np.any(image1 != image2)
        assert np.any(image1 != image3)
        # Test gauss
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.use_filter.setCurrentEnum(NoiseFilterType.Gauss)
        image4 = image_view.viewer_widget.screenshot(flash=False)
        assert np.any(image4 != 255)
        assert np.any(image1 != image4)
        assert np.any(image2 != image4)
        assert np.any(image3 != image4)
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.filter_radius.setValue(1)
        image5 = image_view.viewer_widget.screenshot(flash=False)
        assert np.any(image5 != 255)
        assert np.any(image1 != image5)
        assert np.any(image2 != image5)
        assert np.any(image3 != image5)
        assert np.any(image4 != image5)

    @pytest.mark.windows_ci_skip
    def test_image_view_integration_gauss(self, qtbot, tmp_path, ch_property, image_view):
        def check_parameters(name, index):
            return name == "test" and index == 1

        ch_property.use_filter.setCurrentEnum(NoiseFilterType.Gauss)
        ch_property.filter_radius.setValue(1)
        # Test gauss and fixed range
        ch_property.minimum_value.setValue(100)
        ch_property.maximum_value.setValue(10000)
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(True)
        image_view.viewer_widget.screenshot(flash=False)
        image1 = image_view.viewer_widget._render()
        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.minimum_value.setValue(10)
        image2 = image_view.viewer_widget.screenshot()
        assert np.any(image2 != 255)
        assert np.any(image1 != image2)

        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.maximum_value.setValue(11000)
        image3 = image_view.viewer_widget.screenshot()
        assert np.any(image3 != 255)
        assert np.any(image2 != image3)
        assert np.any(image1 != image3)

        with qtbot.waitSignal(image_view.channel_control.coloring_update), qtbot.waitSignal(
            image_view.channel_control.change_channel, check_params_cb=check_parameters
        ):
            ch_property.fixed.setChecked(False)

        image1 = image_view.viewer_widget.screenshot()
        assert np.any(image1 != 255)
        assert np.any(image1 != image2)
        assert np.any(image1 != image3)
