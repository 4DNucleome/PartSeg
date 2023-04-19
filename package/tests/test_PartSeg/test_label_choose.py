import os

import numpy as np
import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QFileDialog

from PartSeg.common_backend.base_settings import LabelColorDict, ViewSettings
from PartSeg.common_gui.label_create import LabelChoose, LabelEditor, _LabelShow
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
        with pytest.raises(ValueError, match="Cannot delete base item"):
            del dkt["default"]
        assert "default" in dkt

    def test_add_element(self):
        dkt = LabelColorDict({})
        labels = [[1, 2, 3], [120, 230, 100]]
        with pytest.raises(ValueError, match="Cannot write base item"):
            dkt["default"] = labels
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

    def test_add(self, qtbot, base_settings):
        settings = ViewSettings()
        widget = LabelEditor(settings)
        qtbot.addWidget(widget)
        base_count = len(settings.label_color_dict)
        widget.save()
        assert len(settings.label_color_dict) == base_count
        widget.add_color()
        widget.save()
        assert len(settings.label_color_dict) == base_count + 1

    def test_color(self, qtbot, base_settings):
        settings = ViewSettings()
        widget = LabelEditor(settings)
        qtbot.addWidget(widget)
        widget.color_picker.setCurrentColor(QColor(100, 200, 0))
        widget.add_color()
        assert widget.get_colors() == [[100, 200, 0]]

    def test_label_creator_save(self, qtbot, tmp_path, monkeypatch, base_settings):
        widget = LabelEditor(base_settings)
        qtbot.addWidget(widget)
        labels = [[128, 50, 200], [255, 0, 0], [0, 0, 255]]
        widget.set_colors("test", labels)

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

        widget.set_colors("", [])

        widget._import_action()
        assert widget.get_colors() == labels


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


class TestLabelChoose:
    def test_basic(self, qtbot, base_settings):
        widget = LabelChoose(base_settings)
        qtbot.addWidget(widget)
        widget.refresh()

        assert widget.layout().count() == 2

        base_settings.label_color_dict["test"] = [[255, 255, 255], [100, 100, 100]]
        widget.refresh()
        assert widget.layout().count() == 3

        widget.remove("test")
        assert widget.layout().count() == 2

        assert "test" not in base_settings.label_color_dict
