from math import isclose

from pytestqt.qtbot import QtBot
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor

import PartSegData
from PartSeg.common_backend.base_settings import BaseSettings, ColormapDict, ViewSettings
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.colormap_creator import (
    ColormapCreator,
    ColormapEdit,
    ColormapList,
    PColormapCreator,
    PColormapList,
    color_from_qcolor,
    qcolor_from_color,
)
from PartSeg.common_gui.napari_image_view import ImageView
from PartSegCore.color_image import Color, ColorPosition
from PartSegCore.color_image.base_colors import starting_colors
from PartSegImage import TiffImageReader


def test_color_conversion():
    color1 = QColor(70, 52, 190)
    color2 = color_from_qcolor(color1)
    assert qcolor_from_color(color2) == color1
    assert color_from_qcolor(qcolor_from_color(color2)) == color2


class TestColormapEdit:
    def test_click(self, qtbot: QtBot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        width = widget.width() - 20
        with qtbot.waitSignal(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))

        pos = 20 / width
        widget.add_color(ColorPosition(pos, Color(125, 231, 21)))
        assert len(widget.colormap) == 1
        with qtbot.assertNotEmitted(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))
        assert len(widget.colormap) == 0

    def test_distribute_evenly(self, qtbot: QtBot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        widget.add_color(ColorPosition(0.1, Color(125, 231, 21)))
        widget.add_color(ColorPosition(0.23, Color(24, 10, 201)))
        widget.add_color(ColorPosition(0.84, Color(223, 0, 51)))
        assert len(widget.colormap) == 3
        widget.distribute_evenly()
        assert len(widget.colormap) == 3
        assert widget.colormap[0].color_position == 0
        assert widget.colormap[1].color_position == 0.5
        assert widget.colormap[2].color_position == 1
        widget.clear()
        assert len(widget.colormap) == 0

    def test_move(self, qtbot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        width = widget.width() - 20
        pos = 20 / width
        widget.add_color(ColorPosition(pos, Color(125, 231, 21)))
        assert widget.colormap[0].color_position == pos
        qtbot.mousePress(widget, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))
        pos2 = 150 / width
        qtbot.mouseMove(widget, QPoint(70, widget.height() // 2))
        qtbot.mouseMove(widget, QPoint(160, widget.height() // 2))
        qtbot.mouseRelease(widget, Qt.LeftButton, pos=QPoint(160, widget.height() // 2))
        assert widget.colormap[0].color_position == pos2


class TestColormapCreator:
    def test_add_color(self, qtbot: QtBot):
        widget = ColormapCreator()
        colormap_edit = widget.show_colormap
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))
        assert len(widget.current_colormap()) == 1
        assert widget.current_colormap()[0].color == Color(10, 40, 12)
        assert isclose(widget.current_colormap()[0].color_position, 20 / (colormap_edit.width() - 20))
        widget.color_picker.setCurrentColor(color2)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(80, widget.height() // 2))
        assert len(widget.current_colormap()) == 2
        assert widget.current_colormap()[0].color == Color(10, 40, 12)
        assert widget.current_colormap()[1].color == Color(100, 4, 220)
        assert isclose(widget.current_colormap()[0].color_position, 20 / (colormap_edit.width() - 20))
        assert isclose(widget.current_colormap()[1].color_position, 70 / (colormap_edit.width() - 20))

    def test_save(self, qtbot):
        widget = ColormapCreator()
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        widget.add_color(0.1)
        widget.color_picker.setCurrentColor(color2)
        widget.add_color(0.8)

        def check_res(colormap):
            return (
                len(colormap) == 2
                and colormap[0].color == color_from_qcolor(color1)
                and colormap[1].color == color_from_qcolor(color2)
                and colormap[0].color_position == 0.1
                and colormap[1].color_position == 0.8
            )

        with qtbot.wait_signal(widget.colormap_selected, check_params_cb=check_res):
            widget.save_btn.click()

    def test_distribute(self, qtbot):
        widget = ColormapCreator()
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        color3 = QColor(100, 24, 220)
        color4 = QColor(100, 134, 22)
        color5 = QColor(100, 134, 122)
        widget.color_picker.setCurrentColor(color1)
        widget.add_color(0.1)
        widget.color_picker.setCurrentColor(color2)
        widget.add_color(0.3)
        widget.color_picker.setCurrentColor(color3)
        widget.add_color(0.6)
        widget.color_picker.setCurrentColor(color4)
        widget.add_color(0.9)
        widget.color_picker.setCurrentColor(color5)
        widget.add_color(0.95)
        li = widget.current_colormap()
        for el, pos, col in zip(
            li,
            [0.1, 0.3, 0.6, 0.9, 0.95],
            [
                color_from_qcolor(color1),
                color_from_qcolor(color2),
                color_from_qcolor(color3),
                color_from_qcolor(color4),
                color_from_qcolor(color5),
            ],
        ):
            assert el.color_position == pos
            assert el.color == col

        widget.distribute_btn.click()
        li = widget.current_colormap()
        for el, pos, col in zip(
            li,
            [0, 0.25, 0.5, 0.75, 1],
            [
                color_from_qcolor(color1),
                color_from_qcolor(color2),
                color_from_qcolor(color3),
                color_from_qcolor(color4),
                color_from_qcolor(color5),
            ],
        ):
            assert el.color_position == pos
            assert el.color == col


class TestPColormapCreator:
    def test_save(self, qtbot):
        settings = ViewSettings()
        widget = PColormapCreator(settings)
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        widget.add_color(0.1)
        widget.color_picker.setCurrentColor(color2)
        widget.add_color(0.8)

        def check_res(colormap):
            return (
                len(colormap) == 2
                and colormap[0].color == color_from_qcolor(color1)
                and colormap[1].color == color_from_qcolor(color2)
                and colormap[0].color_position == 0.1
                and colormap[1].color_position == 0.8
            )

        with qtbot.wait_signal(widget.colormap_selected, check_params_cb=check_res):
            widget.save_btn.click()

        cmap_dict = settings.get_from_profile("custom_colormap")
        assert len(cmap_dict) == 1
        assert check_res(list(cmap_dict.values())[0])

    def test_custom_colors_save(self, qtbot):
        settings = ViewSettings()
        widget = PColormapCreator(settings)
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCustomColor(2, color1)
        widget.color_picker.setCustomColor(7, color2)
        widget._save_custom_colors()
        custom_color_list = settings.get_from_profile("custom_colors")
        assert custom_color_list[2] == color_from_qcolor(color1)
        assert custom_color_list[7] == color_from_qcolor(color2)
        widget2 = PColormapCreator(settings)
        assert widget2.color_picker.customColor(2) == color1
        assert widget2.color_picker.customColor(7) == color2


class TestColormapList:
    @staticmethod
    def verify_visibility(name, state):
        def _check(_name, _state):
            return name == _name and state == _state

        return _check

    def test_base(self, qtbot):
        dkt = ColormapDict({})
        widget = ColormapList(dkt, starting_colors)
        widget.refresh()
        qtbot.addWidget(widget)
        with qtbot.waitSignal(
            widget.visibility_colormap_change, check_params_cb=self.verify_visibility(starting_colors[0], False)
        ):
            widget.set_state(starting_colors[0], False)
        assert len(widget.get_selected()) == len(starting_colors) - 1
        with qtbot.waitSignal(
            widget.visibility_colormap_change, check_params_cb=self.verify_visibility(starting_colors[0], True)
        ):
            widget.set_state(starting_colors[0], True)
        assert len(widget.get_selected()) == len(starting_colors)
        name = starting_colors[0] + "_reversed"
        with qtbot.waitSignal(widget.visibility_colormap_change, check_params_cb=self.verify_visibility(name, True)):
            widget.set_state(name, True)
        assert len(widget.get_selected()) == len(starting_colors) + 1

    def test_edit_button(self, qtbot):
        dkt = ColormapDict({})
        color_list = ColormapList(dkt, starting_colors)
        color_list.refresh()
        qtbot.addWidget(color_list)
        color_edit = ColormapCreator()
        color_list.edit_signal.connect(color_edit.set_colormap)
        qtbot.addWidget(color_edit)
        cmap = dkt["BlackRed"][0]
        assert cmap != color_edit.current_colormap()
        color_list.get_colormap_widget("BlackRed").edit_btn.click()
        assert cmap == color_edit.current_colormap()
        assert color_list._widget_dict["magma"].edit_btn.isEnabled() is False

    def test_settings_integration(self, qtbot):
        settings = ViewSettings()
        color_list = PColormapList(settings, [])
        color_list.refresh()
        qtbot.addWidget(color_list)
        selected = settings.chosen_colormap[:]
        color_list.set_state("BlackRed", False)
        selected2 = settings.chosen_colormap[:]
        assert len(selected2) + 1 == len(selected)
        assert "BlackRed" not in selected2
        assert "BlackRed" in selected

    def test_image_view_integration(self, qtbot, tmp_path):
        settings = BaseSettings(tmp_path)
        channel_property = ChannelProperty(settings, "test")
        image_view = ImageView(settings, channel_property, "test")
        qtbot.addWidget(channel_property)
        qtbot.addWidget(image_view)
        color_list = PColormapList(settings, ["test"])
        qtbot.addWidget(color_list)
        image = TiffImageReader.read_image(PartSegData.segmentation_analysis_default_image)
        with qtbot.wait_signal(image_view.image_added, timeout=10 ** 6):
            settings.image = image
        color_list.refresh()
        assert image_view.channel_control.channels_count == image.channels
        assert len(color_list.blocked()) == image.channels
        block_count = 0
        for el in settings.colormap_dict.keys():
            widget = color_list.get_colormap_widget(el)
            assert widget.is_checked or el not in starting_colors
            if not widget.checked.isEnabled():
                block_count += 1
        assert block_count == image.channels
        image_view.channel_control.change_selected_color(0, "Grayscale")
        assert len(color_list.blocked()) == image.channels
        assert "Grayscale" in color_list.blocked()
        color_list.refresh()
        assert color_list.get_colormap_widget("Grayscale").checked.isEnabled() is False
        # this lines test if after refresh of widget checkbox stays checkable
        block_count = 0
        for el in settings.colormap_dict.keys():
            widget = color_list.get_colormap_widget(el)
            assert widget.is_checked or el not in starting_colors
            if not widget.checked.isEnabled():
                block_count += 1
        assert block_count == image.channels
