import napari
import numpy as np
import packaging.version
import pytest

from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog
from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.custom_save_dialog import SaveDialog
from PartSeg.plugins.napari_widgets import ROIAnalysisExtraction, ROIMaskExtraction, _settings
from PartSeg.plugins.napari_widgets.roi_extraction_algorithms import ProfilePreviewDialog, QInputDialog
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.load_functions import LoadProfileFromJSON
from PartSegCore.analysis.save_functions import SaveProfilesToJSON
from PartSegCore.mask.algorithm_description import mask_algorithm_dict
from PartSegCore.segmentation import ROIExtractionResult

napari_skip = pytest.mark.skipif(
    packaging.version.parse(napari.__version__) < packaging.version.parse("0.4.10"), reason="To old napari"
)


@pytest.fixture(autouse=True)
def clean_settings(tmp_path):
    old_settings = _settings._settings
    _settings._settings = BaseSettings(tmp_path)
    yield
    _settings._settings = old_settings


def no_action(*_):  # skipcq: PTC-W0049
    pass


@napari_skip
@pytest.mark.parametrize("widget_class", [ROIAnalysisExtraction, ROIMaskExtraction])
def test_extraction_widget(make_napari_viewer, widget_class, monkeypatch, qtbot):

    viewer = make_napari_viewer()
    viewer.add_image(np.ones((10, 10)))
    widget = widget_class(napari_viewer=viewer)
    viewer.window.add_dock_widget(widget)
    widget.reset_choices()
    widget.update_image()
    viewer.add_image(np.ones((10, 10)))
    monkeypatch.setattr(QInputDialog, "getText", get_text_mock("prof3"))
    monkeypatch.setattr(ProfilePreviewDialog, "exec_", no_action)
    assert widget.profile_combo_box.count() == 1
    widget.save_action()
    with qtbot.waitSignal(widget.algorithm_chose.algorithm_choose.currentIndexChanged):
        widget.algorithm_chose.algorithm_choose.setCurrentIndex(1)
    with qtbot.waitSignal(widget.profile_combo_box.currentIndexChanged):
        widget.profile_combo_box.setCurrentIndex(1)
    assert widget.profile_combo_box.count() == 2
    widget.refresh_profiles()
    assert widget.profile_combo_box.count() == 2
    widget.manage_action()
    widget.select_profile("prof3")
    widget.update_image()

    res = ROIExtractionResult(np.ones((10, 10), dtype=np.uint8), widget.profile_dict["prof3"])
    res.roi[0, 0] = 2
    res.roi[1, 1] = 0

    widget.set_result(res)
    widget.set_result(res)

    curr_widget = widget.algorithm_chose.current_widget()
    curr_widget.form_widget.widgets_dict["mask"]._widget.value = viewer.layers["ROI"]
    widget.update_mask()


def get_text_mock(text):
    def func(*_args, **_kwargs):
        return text, True

    return func


@pytest.mark.parametrize("register", [analysis_algorithm_dict, mask_algorithm_dict])
def test_profile_preview_dialog(part_settings, register, qtbot, monkeypatch, tmp_path):
    elem_name = next(iter(register))
    profiles = {
        "prof1": ROIExtractionProfile("prof1", elem_name, register[elem_name].get_default_values()),
        "prof2": ROIExtractionProfile("prof2", elem_name, register[elem_name].get_default_values()),
    }
    dialog = ProfilePreviewDialog(profiles, register, part_settings)
    qtbot.add_widget(dialog)
    assert dialog.profile_view.toPlainText() == ""
    assert dialog.profile_list.count() == 2
    with qtbot.waitSignal(dialog.profile_list.currentTextChanged):
        dialog.profile_list.setCurrentRow(0)
    assert dialog.profile_list.currentItem().text() == "prof1"
    assert dialog.profile_view.toPlainText() != ""
    monkeypatch.setattr(QInputDialog, "getText", get_text_mock("prof3"))
    monkeypatch.setattr(SaveDialog, "exec_", lambda x: True)
    monkeypatch.setattr(
        SaveDialog,
        "get_result",
        lambda x: (tmp_path / "profile.json", None, SaveProfilesToJSON, SaveProfilesToJSON.get_default_values()),
    )
    monkeypatch.setattr(ExportDialog, "exec_", lambda x: True)
    monkeypatch.setattr(ExportDialog, "get_export_list", lambda x: ["prof1"])
    dialog.export_action()
    dialog.rename_action()
    assert "prof3" in profiles
    assert len(profiles) == 2
    with qtbot.waitSignal(dialog.profile_list.currentTextChanged):
        dialog.profile_list.setCurrentRow(0)
    assert dialog.profile_list.currentItem().text() == "prof2"

    dialog.delete_action()
    assert len(profiles) == 1
    with qtbot.waitSignal(dialog.profile_list.currentTextChanged):
        dialog.profile_list.setCurrentRow(0)

    monkeypatch.setattr(ImportDialog, "exec_", lambda x: True)
    monkeypatch.setattr(ImportDialog, "get_import_list", lambda x: [("prof1", "prof1")])
    monkeypatch.setattr(CustomLoadDialog, "exec_", lambda x: True)
    monkeypatch.setattr(
        CustomLoadDialog, "get_result", lambda x: ([tmp_path / "profile.json"], None, LoadProfileFromJSON)
    )

    dialog.import_action()
    assert len(profiles) == 2
    assert dialog.profile_list.count() == 2


@napari_skip
def test_measurement_create(make_napari_viewer):
    from PartSeg.plugins.napari_widgets.measurement_widget import SimpleMeasurement

    viewer = make_napari_viewer()
    measurement = SimpleMeasurement(viewer)
    measurement.reset_choices()
