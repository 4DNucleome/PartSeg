import os

import numpy as np
from napari.utils import Colormap
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFileDialog

import PartSegData
from PartSeg.common_backend.base_settings import BaseSettings, ColormapDict, ViewSettings
from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.colormap_creator import (
    ColormapCreator,
    ColormapEdit,
    ColormapList,
    PColormapCreator,
    PColormapCreatorMid,
    PColormapList,
    color_from_qcolor,
    qcolor_from_color,
)
from PartSeg.common_gui.napari_image_view import ImageView
from PartSegCore.color_image.base_colors import Color, starting_colors
from PartSegImage import TiffImageReader


def test_color_conversion():
    color1 = QColor(70, 52, 190)
    color2 = color_from_qcolor(color1)
    assert qcolor_from_color(color2) == color1
    assert color_from_qcolor(qcolor_from_color(color2)) == color2


class TestColormapEdit:
    def test_click(self, qtbot: QtBot, qapp):
        widget = ColormapEdit()
        widget.show()
        qtbot.addWidget(widget)
        width = widget.width() - 20
        with qtbot.waitSignal(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))

        pos = 20 / width
        widget.add_color(pos, Color(red=0.274, green=0.02, blue=0.745))
        widget.add_color(pos / 2, Color(red=0.274, green=0.2, blue=0.745))
        assert len(widget.colormap.colors) == 4
        with qtbot.assertNotEmitted(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))
        assert len(widget.colormap.colors) == 3
        qapp.processEvents()
        widget.hide()

    def test_distribute_evenly(self, qtbot: QtBot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        widget.add_color(0.1, Color(red=0.49, green=0.9, blue=0.08))
        widget.add_color(0.23, Color(red=0.09, green=0.03, blue=0.788))
        widget.add_color(0.84, Color(red=0.874, green=0, blue=0.2))
        assert len(widget.colormap.colors) == 5
        widget.distribute_evenly()
        assert len(widget.colormap.colors) == 3
        assert widget.colormap.controls[0] == 0
        assert widget.colormap.controls[1] == 0.5
        assert widget.colormap.controls[2] == 1
        widget.clear()
        assert len(widget.colormap.colors) == 1

    def test_move(self, qtbot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        width = widget.width() - 20
        pos = 20 / width
        widget.add_color(pos, Color(red=0.49, green=0.9, blue=0.08))
        assert np.isclose(widget.colormap.controls[1], pos)
        qtbot.mousePress(widget, Qt.MouseButton.LeftButton, pos=QPoint(30, widget.height() // 2))
        pos2 = 150 / width
        qtbot.mouseMove(widget, QPoint(70, widget.height() // 2))
        qtbot.mouseMove(widget, QPoint(160, widget.height() // 2))
        qtbot.mouseRelease(widget, Qt.MouseButton.LeftButton, pos=QPoint(160, widget.height() // 2))
        assert np.isclose(widget.colormap.controls[1], pos2)


class TestColormapCreator:
    def test_add_color(self, qtbot: QtBot):
        widget = ColormapCreator()
        colormap_edit = widget.show_colormap
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(30, widget.height() // 2))
        assert len(widget.current_colormap().colors) == 3
        assert np.allclose(
            widget.current_colormap().colors[0], Color(red=10 / 255, green=40 / 255, blue=12 / 255).as_tuple()
        )
        assert np.isclose(widget.current_colormap().controls[1], 20 / (colormap_edit.width() - 20))
        widget.color_picker.setCurrentColor(color2)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(80, widget.height() // 2))
        assert len(widget.current_colormap().colors) == 4
        assert np.allclose(
            widget.current_colormap().colors[1], Color(red=10 / 255, green=40 / 255, blue=12 / 255).as_tuple()
        )
        assert np.allclose(
            widget.current_colormap().colors[2], Color(red=100 / 255, green=4 / 255, blue=220 / 255).as_tuple()
        )
        assert np.isclose(widget.current_colormap().controls[1], 20 / (colormap_edit.width() - 20))
        assert np.isclose(widget.current_colormap().controls[2], 70 / (colormap_edit.width() - 20))

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
                len(colormap.colors) == 4
                and np.allclose(colormap.colors[1], color_from_qcolor(color1).as_tuple())
                and np.allclose(colormap.colors[2], color_from_qcolor(color2).as_tuple())
                and np.isclose(colormap.controls[1], 0.1)
                and np.isclose(colormap.controls[2], 0.8)
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
        assert np.allclose(li.controls, [0, 0.1, 0.3, 0.6, 0.9, 0.95, 1])
        for el, col in zip(
            li.colors,
            [
                color_from_qcolor(color1),
                color_from_qcolor(color1),
                color_from_qcolor(color2),
                color_from_qcolor(color3),
                color_from_qcolor(color4),
                color_from_qcolor(color5),
                color_from_qcolor(color5),
            ],
        ):
            assert np.allclose(el, col.as_tuple())

        widget.distribute_btn.click()
        li = widget.current_colormap()
        assert np.allclose(li.controls, [0, 0.25, 0.5, 0.75, 1])
        for el, col in zip(
            li.colors,
            [
                color_from_qcolor(color1),
                color_from_qcolor(color2),
                color_from_qcolor(color3),
                color_from_qcolor(color4),
                color_from_qcolor(color5),
            ],
        ):
            assert np.allclose(el, col.as_tuple())

    def test_colormap_creator_save(self, qtbot, tmp_path, monkeypatch, base_settings):
        widget = PColormapCreatorMid(base_settings)
        cmap = Colormap([[0, 0, 0], [0.5, 0.5, 0.5], [1, 1, 1]], controls=[0, 0.3, 1])
        widget.show_colormap.colormap = cmap
        qtbot.addWidget(widget)
        target_path = str(tmp_path / "test.json")

        def exec_(self_):
            self_.selectFile(target_path)
            self_.accept()
            return True

        def selected_name_filter(self_):
            return self_.nameFilters()[0]

        def selected_files(self_):
            return [target_path]

        monkeypatch.setattr(QFileDialog, "exec_", exec_)
        monkeypatch.setattr(QFileDialog, "selectedNameFilter", selected_name_filter)
        monkeypatch.setattr(QFileDialog, "selectedFiles", selected_files)

        widget._export_action()
        assert os.path.isfile(target_path)

        widget.clear_btn.click()
        assert widget.show_colormap.colormap != cmap

        widget._import_action()

        assert widget.show_colormap.colormap == cmap


class TestPColormapCreator:
    def test_save(self, qtbot, base_settings):
        widget = PColormapCreator(base_settings)
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        widget.add_color(0.1)
        widget.color_picker.setCurrentColor(color2)
        widget.add_color(0.8)

        def check_res(colormap):
            return (
                len(colormap.colors) == 4
                and np.allclose(colormap.colors[0], color_from_qcolor(color1).as_tuple())
                and np.allclose(colormap.colors[1], color_from_qcolor(color1).as_tuple())
                and np.allclose(colormap.colors[2], color_from_qcolor(color2).as_tuple())
                and np.allclose(colormap.colors[3], color_from_qcolor(color2).as_tuple())
                and np.allclose(colormap.controls, [0, 0.1, 0.8, 1])
            )

        with qtbot.wait_signal(widget.colormap_selected, check_params_cb=check_res):
            widget.save_btn.click()

        cmap_dict = base_settings.get_from_profile("custom_colormap")
        assert len(cmap_dict) == 1
        assert check_res(next(iter(cmap_dict.values())))

    def test_custom_colors_save(self, qtbot, base_settings):
        widget = PColormapCreator(base_settings)
        qtbot.addWidget(widget)
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCustomColor(2, color1)
        widget.color_picker.setCustomColor(7, color2)
        widget._save_custom_colors()
        custom_color_list = base_settings.get_from_profile("custom_colors")
        assert custom_color_list[2] == color_from_qcolor(color1)
        assert custom_color_list[7] == color_from_qcolor(color2)
        widget2 = PColormapCreator(base_settings)
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
        name = f"{starting_colors[0]}_reversed"
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
        cmap = dkt["red"][0]
        assert np.any(cmap.colors != color_edit.current_colormap().colors)
        color_list.get_colormap_widget("red").edit_btn.click()
        assert np.all(cmap.colors == color_edit.current_colormap().colors)
        assert color_list._widget_dict["magma"].edit_btn.isEnabled() is False

    def test_settings_integration(self, qtbot):
        settings = ViewSettings()
        color_list = PColormapList(settings, [])
        color_list.refresh()
        qtbot.addWidget(color_list)
        selected = settings.chosen_colormap[:]
        color_list.set_state("red", False)
        selected2 = settings.chosen_colormap[:]
        assert len(selected2) + 1 == len(selected)
        assert "red" not in selected2
        assert "red" in selected

    def test_image_view_integration(self, qtbot, tmp_path):
        settings = BaseSettings(tmp_path)
        channel_property = ChannelProperty(settings, "test")
        image_view = ImageView(settings, channel_property, "test")
        qtbot.addWidget(channel_property)
        qtbot.addWidget(image_view)
        color_list = PColormapList(settings, ["test"])
        qtbot.addWidget(color_list)
        image = TiffImageReader.read_image(PartSegData.segmentation_analysis_default_image)
        with qtbot.wait_signal(image_view.image_added, timeout=10**6):
            settings.image = image
        color_list.refresh()
        assert image_view.channel_control.channels_count == image.channels
        assert len(color_list.blocked()) == image.channels
        block_count = 0
        for el in settings.colormap_dict:
            widget = color_list.get_colormap_widget(el)
            assert widget.is_checked or el not in starting_colors
            if not widget.checked.isEnabled():
                block_count += 1
        assert block_count == image.channels
        image_view.channel_control.change_selected_color(0, "gray")
        assert len(color_list.blocked()) == image.channels
        assert "gray" in color_list.blocked()
        color_list.refresh()
        assert color_list.get_colormap_widget("gray").checked.isEnabled() is False
        # this lines test if after refresh of widget checkbox stays checkable
        block_count = 0
        for el in settings.colormap_dict:
            widget = color_list.get_colormap_widget(el)
            assert widget.is_checked or el not in starting_colors
            if not widget.checked.isEnabled():
                block_count += 1
        assert block_count == image.channels
