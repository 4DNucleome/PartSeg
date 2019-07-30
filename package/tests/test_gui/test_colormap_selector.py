from math import isclose
from pytestqt.qtbot import QtBot
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QColor

from PartSeg.common_gui.colormap_creator import color_from_qcolor, qcolor_from_color, ColormapEdit, ColormapCreator, \
    PColormapCreator
from PartSeg.project_utils_qt.settings import ViewSettings
from PartSeg.utils.color_image import ColorPosition, Color


def test_color_conversion():
    color1 = QColor(70, 52, 190)
    color2 = color_from_qcolor(color1)
    assert qcolor_from_color(color2) == color1
    assert color_from_qcolor(qcolor_from_color(color2)) == color2


class TestColormapEdit:
    def test_click(self, qtbot: QtBot):
        widget = ColormapEdit()
        qtbot.addWidget(widget)
        width = widget.width()-20
        with qtbot.waitSignal(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height()/2))

        pos = 20/width
        widget.add_color(ColorPosition(pos, Color(125, 231, 21)))
        assert len(widget.colormap) == 1
        with qtbot.assertNotEmitted(widget.double_clicked):
            qtbot.mouseDClick(widget, Qt.LeftButton, pos=QPoint(30, widget.height()/2))
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
        qtbot.mousePress(widget, Qt.LeftButton, pos=QPoint(30, widget.height()/2))
        pos2 = 150 / width
        qtbot.mouseMove(widget, QPoint(70, widget.height() / 2))
        qtbot.mouseMove(widget, QPoint(160, widget.height()/2))
        qtbot.mouseRelease(widget, Qt.LeftButton, pos=QPoint(160, widget.height()/2))
        assert widget.colormap[0].color_position == pos2


class TestColormapCreator:
    def test_add_color(self, qtbot: QtBot):
        widget = ColormapCreator()
        colormap_edit = widget.show_colormap
        color1 = QColor(10, 40, 12)
        color2 = QColor(100, 4, 220)
        widget.color_picker.setCurrentColor(color1)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(30, widget.height()/2))
        assert len(widget.current_colormap()) == 1
        assert widget.current_colormap()[0].color == Color(10, 40, 12)
        assert isclose(widget.current_colormap()[0].color_position, 20/(colormap_edit.width() - 20))
        widget.color_picker.setCurrentColor(color2)
        with qtbot.waitSignal(colormap_edit.double_clicked):
            qtbot.mouseDClick(colormap_edit, Qt.LeftButton, pos=QPoint(80, widget.height()/2))
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
            return len(colormap) == 2 and colormap[0].color == color_from_qcolor(color1) and \
                   colormap[1].color == color_from_qcolor(color2) and colormap[0].color_position == 0.1 and \
                   colormap[1].color_position == 0.8

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
        for el, pos, col in zip(li, [0.1, 0.3, 0.6, 0.9, 0.95],
                                [color_from_qcolor(color1), color_from_qcolor(color2), color_from_qcolor(color3),
                                 color_from_qcolor(color4), color_from_qcolor(color5)]):
            assert el.color_position == pos
            assert el.color == col

        widget.distribute_btn.click()
        li = widget.current_colormap()
        for el, pos, col in zip(li, [0, 0.25, 0.5, 0.75, 1],
                                [color_from_qcolor(color1), color_from_qcolor(color2), color_from_qcolor(color3),
                                 color_from_qcolor(color4), color_from_qcolor(color5)]):
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
            return len(colormap) == 2 and colormap[0].color == color_from_qcolor(color1) and \
                   colormap[1].color == color_from_qcolor(color2) and colormap[0].color_position == 0.1 and \
                   colormap[1].color_position == 0.8

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
