# pylint: disable=R0201
import platform
from unittest.mock import MagicMock

import numpy as np
import pytest
import qtpy
from napari.layers import Image as NapariImage
from qtpy.QtCore import QPoint
from test_PartSeg.utils import CI_BUILD
from vispy.geometry import Rect

from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.napari_image_view import (
    ORDER_DICT,
    ImageInfo,
    ImageView,
    QMenu,
    SearchComponentModal,
    SearchType,
    _print_dict,
)
from PartSegCore.roi_info import ROIInfo
from PartSegImage import Image

pyside_skip = pytest.mark.skipif(qtpy.API_NAME == "PySide2", reason="PySide2 problem with mocking excec_")


def test_image_info():
    image_info = ImageInfo(Image(np.zeros((10, 10)), image_spacing=(1, 1), axes_order="XY"), [])
    assert not image_info.coords_in([1, 1])
    assert np.all(image_info.translated_coords([1, 1]) == [1, 1])

    image_info.layers.append(NapariImage(image_info.image.get_channel(0), scale=(1, 1, 10, 10)))
    assert image_info.coords_in([0.5, 0.5, 1, 1])
    assert np.all(image_info.translated_coords([1, 1, 1, 1]) == [1, 1, 1, 1])


def test_print_dict():
    dkt = {"a": 1, "b": {"e": 1, "d": [1, 2, 4]}}
    res = _print_dict(dkt)
    lines = res.split("\n")
    assert len(lines) == 4
    assert lines[0].startswith("a")
    assert lines[2].startswith("  e")


@pytest.fixture
def image_view(base_settings, image2, qtbot):
    ch_prop = ChannelProperty(base_settings, "test")
    view = ImageView(base_settings, channel_property=ch_prop, name="test")
    qtbot.addWidget(ch_prop)
    qtbot.addWidget(view)
    base_settings.image = image2
    return view


class TestImageView:
    def test_constructor(self, base_settings, qtbot):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)

    def test_print_info(self, base_settings, image2, qtbot, monkeypatch):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        with qtbot.assert_not_emitted(view.text_info_change):
            view.print_info()
        base_settings.image = image2
        monkeypatch.setattr(view, "_coordinates", lambda: (-10, 0.5, 0.5, 0.5))
        with qtbot.waitSignal(view.text_info_change, check_params_cb=lambda x: x == ""):
            view.print_info()
        base_settings.roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        monkeypatch.setattr(view, "_coordinates", lambda: (0.5, 0.5, 0.5, 0.5))
        with qtbot.waitSignal(view.text_info_change, check_params_cb=lambda x: "component: " in x):
            view.print_info()

    def test_synchronize_viewers(self, base_settings, image2, qtbot):
        ch_prop = ChannelProperty(base_settings, "test")
        view1 = ImageView(base_settings, channel_property=ch_prop, name="test")
        view2 = ImageView(base_settings, channel_property=ch_prop, name="test2")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view1)
        qtbot.addWidget(view2)
        base_settings.image = image2

        view1.viewer.dims.set_point(1, 5)
        view2.set_state(view1.get_state())
        view1.viewer.dims.ndisplay = 3
        view2.set_state(view1.get_state())

    def test_order_reset(self, image_view):
        image_view._rotate_dim()
        assert image_view.viewer.dims.order == (0, 2, 1, 3)
        image_view._reset_view()
        assert image_view.viewer.dims.order == (0, 1, 2, 3)

    def test_update_rendering(self, base_settings, image_view):
        base_settings.roi = np.ones(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        image_view.update_rendering()

    def test_roi_rendering(self, base_settings, image_view, tmp_path):
        roi = np.ones(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 1, 1] = 0
        image_view.set_roi(ROIInfo(None))
        base_settings.roi = roi
        image_view.update_roi_border()
        assert np.any(image_view.image_info[str(tmp_path / "test2.tiff")].roi.data == 1)
        image_view.settings.set_in_profile(f"{image_view.name}.image_state.only_border", False)
        image_view.update_roi_border()
        assert np.count_nonzero(image_view.image_info[str(tmp_path / "test2.tiff")].roi.data) == 20 ** 3 - 20
        base_settings.roi = np.ones(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].roi.data)
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].roi.scale == (1, 10 ** 6, 10 ** 6, 10 ** 6))
        base_settings.image.set_spacing((10 ** -4,) * 3)
        image_view.update_spacing_info()
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].roi.scale == (1, 10 ** 5, 10 ** 5, 10 ** 5))
        base_settings.roi = None
        assert not image_view.image_info[str(tmp_path / "test2.tiff")].roi.visible
        base_settings.roi = roi
        assert image_view.image_info[str(tmp_path / "test2.tiff")].roi.visible
        image_view.remove_all_roi()
        assert not image_view.image_info[str(tmp_path / "test2.tiff")].roi.visible

    def test_has_image(self, base_settings, image_view, image, image2):
        base_settings.image = image2
        assert image_view.has_image(image2)
        assert not image_view.has_image(image)
        image_view.update_spacing_info()

    def test_mask_rendering(self, base_settings, image_view, qtbot, tmp_path):
        image_view.set_mask()
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].mask.data == 1)
        # double add for test if proper refresh mask is working
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = np.ones(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].mask.data == 0)
        base_settings.set_in_profile("mask_presentation_opacity", 0.5)
        # view.update_mask_parameters()
        assert image_view.image_info[str(tmp_path / "test2.tiff")].mask.opacity == 0.5
        base_settings.set_in_profile("mask_presentation_color", (255, 0, 0))
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].mask.color[1] == (1, 0, 0, 1))
        base_settings.set_in_profile("mask_presentation_color", (128, 0, 0))
        assert np.allclose(image_view.image_info[str(tmp_path / "test2.tiff")].mask.color[1], (128 / 255, 0, 0, 1))

        assert not image_view.image_info[str(tmp_path / "test2.tiff")].mask.visible
        with qtbot.waitSignal(image_view.mask_chk.stateChanged):
            image_view.mask_chk.setChecked(True)
        assert image_view.image_info[str(tmp_path / "test2.tiff")].mask.visible
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].mask.scale == (1, 10 ** 6, 10 ** 6, 10 ** 6))
        base_settings.image.set_spacing((10 ** -4,) * 3)
        image_view.update_spacing_info()
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].mask.scale == (1, 10 ** 5, 10 ** 5, 10 ** 5))

    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_mask_control_visibility(self, base_settings, image_view, qtbot, tmp_path):
        image_view.show()
        assert not image_view.mask_chk.isVisible()
        image_view.set_mask()
        assert not image_view.mask_chk.isVisible()
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        assert image_view.mask_chk.isVisible()
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = None
        assert not image_view.mask_chk.isVisible()
        image_view.hide()

    def test_points_rendering(self, base_settings, image_view, tmp_path):
        assert image_view.points_layer is None
        base_settings.points = [(0, 5, 5, 5)]
        assert image_view.points_layer is not None
        assert image_view.points_layer.visible
        image_view.toggle_points_visibility()
        assert not image_view.points_layer.visible

    @pytest.mark.skipif((platform.system() == "Windows") and CI_BUILD, reason="glBindFramebuffer with no OpenGL")
    def test_points_button_visibility(self, base_settings, image_view, qtbot, tmp_path):
        image_view.show()
        assert not image_view.points_view_button.isVisible()
        base_settings.points = [(0, 5, 5, 5)]
        assert image_view.points_view_button.isVisible()
        base_settings.points = None
        assert not image_view.points_view_button.isVisible()
        image_view.hide()

    @pyside_skip
    def test_dim_menu(self, base_settings, image_view, monkeypatch):
        called = []

        def check_menu(self, point):
            assert len(self.actions()) == len(ORDER_DICT)
            called.append(1)

        monkeypatch.setattr(QMenu, "exec_", check_menu)
        image_view._dim_order_menu(QPoint(0, 0))
        assert called == [1]

    def test_update_alternatives(self, base_settings, image_view, tmp_path):
        roi = np.ones(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 1, 1] = 0
        base_settings.roi = ROIInfo(roi, alternative={"test": np.ones(roi.shape, dtype=np.uint8) - roi})
        image_view.update_roi_border()
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].roi.data == roi)
        image_view.roi_alternative_selection = "test"
        image_view.update_roi_border()
        assert np.all(image_view.image_info[str(tmp_path / "test2.tiff")].roi.data != roi)

    def test_marking_component(self, base_settings, image_view, tmp_path):
        roi = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 2:-2, 2:-2, 2:-2] = 1
        base_settings.roi = roi
        image_view.component_mark(1, False)
        assert image_view.image_info[str(tmp_path / "test2.tiff")].highlight is not None
        assert image_view.image_info[str(tmp_path / "test2.tiff")].highlight.visible
        assert image_view.image_info[str(tmp_path / "test2.tiff")].highlight in image_view.viewer.layers
        assert "timer" not in image_view.image_info[str(tmp_path / "test2.tiff")].highlight.metadata
        image_view.component_unmark(0)
        assert not image_view.image_info[str(tmp_path / "test2.tiff")].highlight.visible
        image_view.component_mark(10, False)
        assert not image_view.image_info[str(tmp_path / "test2.tiff")].highlight.visible

    @pytest.mark.enablethread
    def test_marking_component_flash(self, base_settings, image_view, tmp_path, qtbot):
        roi = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 2:-2, 2:-2, 2:-2] = 1
        base_settings.roi = roi
        image_view.component_mark(1, True)
        assert image_view.image_info[str(tmp_path / "test2.tiff")].highlight.visible
        assert "timer" in image_view.image_info[str(tmp_path / "test2.tiff")].highlight.metadata
        timer = image_view.image_info[str(tmp_path / "test2.tiff")].highlight.metadata["timer"]
        assert timer.isActive()
        qtbot.wait(800)
        image_view.component_unmark(0)
        assert not image_view.image_info[str(tmp_path / "test2.tiff")].highlight.visible
        assert not timer.isActive()

    @pytest.mark.parametrize("pos", [(-(10 ** 8), -(10 ** 8)), (10 ** 8, 10 ** 8)])
    def test_update_camera_position_component_marking(self, base_settings, image_view, pos):
        roi = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 2:-2, 2:-2, 2:-2] = 1
        base_settings.roi = roi
        image_view.viewer.dims.set_point(1, 0)
        rect = Rect(image_view.viewer_widget.view.camera.get_state()["rect"])
        rect.pos = pos
        rect.size = (100, 100)
        image_view.viewer_widget.view.camera.set_state({"rect": rect})
        image_view.component_mark(1, False)
        assert image_view.viewer.dims.point[1] != 0
        assert image_view.viewer_widget.view.camera.get_state()["rect"].pos != pos

    def test_zoom_mark(self, base_settings, image_view):
        roi = np.zeros(base_settings.image.get_channel(0).shape, dtype=np.uint8)
        roi[..., 2:-2, 2:-2, 2:-2] = 1
        base_settings.roi = roi
        image_view.component_zoom(1)
        image_view.component_zoom(10)


def test_search_component_modal(qtbot, image_view, monkeypatch):
    monkeypatch.setattr(image_view, "component_mark", MagicMock())
    monkeypatch.setattr(image_view, "component_zoom", MagicMock())
    monkeypatch.setattr(image_view, "component_unmark", MagicMock())
    modal = SearchComponentModal(image_view, SearchType.Highlight, component_num=1, max_components=5)
    image_view.component_mark.assert_called_with(1, flash=True)
    modal.component_selector.setValue(2)
    image_view.component_mark.assert_called_with(2, flash=True)
    assert image_view.component_zoom.call_count == 0
    modal.zoom_to.setCurrentEnum(SearchType.Zoom_in)
    image_view.component_zoom.assert_called_with(2)
    assert image_view.component_mark.call_count == 2
    modal.component_selector.setValue(1)
    image_view.component_zoom.assert_called_with(1)
    modal.close()
    image_view.component_unmark.assert_called_once()
