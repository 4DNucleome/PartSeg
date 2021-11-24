# pylint: disable=R0201

import numpy as np
import pytest
import qtpy
from napari.layers import Image as NapariImage
from qtpy.QtCore import QPoint

from PartSeg.common_gui.channel_control import ChannelProperty
from PartSeg.common_gui.napari_image_view import ORDER_DICT, ImageInfo, ImageView, QMenu, _print_dict
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

    def test_order_reset(self, base_settings, image2, qtbot):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2

        view._rotate_dim()
        assert view.viewer.dims.order == (0, 2, 1, 3)
        view._reset_view()
        assert view.viewer.dims.order == (0, 1, 2, 3)

    def test_update_rendering(self, base_settings, image2, qtbot):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2
        base_settings.roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        view.update_rendering()

    def test_roi_rendering(self, base_settings, image2, qtbot, tmp_path):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2
        roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        roi[..., 1, 1] = 0
        base_settings.roi = roi
        view.update_roi_border()
        assert np.any(view.image_info[str(tmp_path / "test2.tiff")].roi.data == 1)
        view.settings.set_in_profile(f"{view.name}.image_state.only_border", False)
        view.update_roi_border()
        assert np.count_nonzero(view.image_info[str(tmp_path / "test2.tiff")].roi.data) == 20 ** 3 - 20
        base_settings.roi = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].roi.data)
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].roi.scale == (1, 10 ** 6, 10 ** 6, 10 ** 6))
        image2.set_spacing((10 ** -4,) * 3)
        view.update_spacing_info()
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].roi.scale == (1, 10 ** 5, 10 ** 5, 10 ** 5))
        view.remove_all_roi()
        assert view.image_info[str(tmp_path / "test2.tiff")].roi is None

    def test_has_image(self, base_settings, image, image2, qtbot):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2
        assert view.has_image(image2)
        assert not view.has_image(image)
        view.update_spacing_info()

    def test_mask_rendering(self, base_settings, image2, qtbot, tmp_path):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2
        view.set_mask()
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = np.zeros(image2.get_channel(0).shape, dtype=np.uint8)
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].mask.data == 1)
        # double add for test if proper refresh mask is working
        with qtbot.waitSignal(base_settings.mask_changed):
            base_settings.mask = np.ones(image2.get_channel(0).shape, dtype=np.uint8)
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].mask.data == 0)
        base_settings.set_in_profile("mask_presentation_opacity", 0.5)
        # view.update_mask_parameters()
        assert view.image_info[str(tmp_path / "test2.tiff")].mask.opacity == 0.5
        base_settings.set_in_profile("mask_presentation_color", (255, 0, 0))
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].mask.color[1] == (1, 0, 0, 1))
        base_settings.set_in_profile("mask_presentation_color", (128, 0, 0))
        assert np.allclose(view.image_info[str(tmp_path / "test2.tiff")].mask.color[1], (128 / 255, 0, 0, 1))

        assert not view.image_info[str(tmp_path / "test2.tiff")].mask.visible
        with qtbot.waitSignal(view.mask_chk.stateChanged):
            view.mask_chk.setChecked(True)
        assert view.image_info[str(tmp_path / "test2.tiff")].mask.visible
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].mask.scale == (1, 10 ** 6, 10 ** 6, 10 ** 6))
        image2.set_spacing((10 ** -4,) * 3)
        view.update_spacing_info()
        assert np.all(view.image_info[str(tmp_path / "test2.tiff")].mask.scale == (1, 10 ** 5, 10 ** 5, 10 ** 5))

    def test_points_rendering(self, base_settings, image2, qtbot, tmp_path):
        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2
        assert view.points_layer is None
        base_settings.points = [(0, 5, 5, 5)]
        assert view.points_layer is not None
        assert view.points_layer.visible
        view.toggle_points_visibility()
        assert not view.points_layer.visible

    @pyside_skip
    def test_dim_menu(self, base_settings, image2, qtbot, monkeypatch):
        called = []

        def check_menu(self, point):
            assert len(self.actions()) == len(ORDER_DICT)
            called.append(1)

        ch_prop = ChannelProperty(base_settings, "test")
        view = ImageView(base_settings, channel_property=ch_prop, name="test")
        qtbot.addWidget(ch_prop)
        qtbot.addWidget(view)
        base_settings.image = image2

        monkeypatch.setattr(QMenu, "exec_", check_menu)
        view._dim_order_menu(QPoint(0, 0))
        assert called == [1]
