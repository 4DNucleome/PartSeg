import numpy as np
import pytest
from qtpy.QtGui import QColor

from PartSeg.common_backend.base_settings import LabelColorDict, ViewSettings
from PartSeg.common_gui.label_create import LabelEditor, _LabelShow
from PartSegCore.color_image.color_data import sitk_labels


class TestLabelColorDict:
    def test_has_default(self):
        dkt = LabelColorDict({})
        assert "default" in dkt
        assert isinstance(dkt.get_array("default"), np.ndarray)
        assert dkt.get_array("default").dtype == np.uint8
        assert isinstance(dkt["default"][1], bool)
        assert dkt["default"][1] is False

    def test_delete_default(self):
        dkt = LabelColorDict({})
        with pytest.raises(ValueError):
            del dkt["default"]
        assert "default" in dkt

    def test_add_element(self):
        dkt = LabelColorDict({})
        labels = [[1, 2, 3], [120, 230, 100]]
        with pytest.raises(ValueError):
            dkt["test"] = labels
        assert "test" not in dkt
        dkt["custom_test"] = labels
        assert "custom_test" in dkt
        assert dkt["custom_test"][1] is True
        assert isinstance(dkt.get_array("custom_test"), np.ndarray)
        assert dkt.get_array("custom_test").dtype == np.uint8
        del dkt["custom_test"]
        assert "custom_test" not in dkt


class TestLabelEditor:
    def test_init(self, qtbot):
        settings = ViewSettings()
        widget = LabelEditor(settings)
        qtbot.addWidget(widget)
        assert len(widget.get_colors()) == 0

    def test_add(self, qtbot):
        settings = ViewSettings()
        widget = LabelEditor(settings)
        qtbot.addWidget(widget)
        base_count = len(settings.label_color_dict)
        widget.save()
        assert len(settings.label_color_dict) == base_count
        widget.add_color()
        widget.save()
        assert len(settings.label_color_dict) == base_count + 1

    def test_color(self, qtbot):
        settings = ViewSettings()
        widget = LabelEditor(settings)
        qtbot.addWidget(widget)
        widget.color_picker.setCurrentColor(QColor(100, 200, 0))
        widget.add_color()
        assert widget.get_colors() == [[100, 200, 0]]


def test__label_show(qtbot):
    widget = _LabelShow(np.array([[255, 255, 255], [100, 100, 100]], dtype=np.uint8))
    qtbot.addWidget(widget)
    assert widget.image.height() == 1
    assert widget.image.width() == 2
    widget.set_labels(np.array([[255, 255, 255], [100, 100, 100], [250, 0, 50]], dtype=np.uint8))
    assert widget.image.height() == 1
    assert widget.image.width() == 3
    widget.set_labels(np.array(sitk_labels, dtype=np.uint8))
    assert widget.image.height() == 1
    assert widget.image.width() == len(sitk_labels)
    widget.set_labels(np.array([[255, 255, 255, 255], [100, 100, 100, 255]], dtype=np.uint8))
    assert widget.image.height() == 1
    assert widget.image.width() == 2
