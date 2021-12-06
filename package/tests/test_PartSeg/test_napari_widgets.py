import napari
import numpy as np
import packaging.version
import pytest

from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.custom_save_dialog import CustomSaveDialog
from PartSeg.common_gui.napari_image_view import SearchType
from PartSeg.plugins.napari_widgets import (
    MaskCreateNapari,
    ROIAnalysisExtraction,
    ROIMaskExtraction,
    SearchLabel,
    _settings,
)
from PartSeg.plugins.napari_widgets.roi_extraction_algorithms import ProfilePreviewDialog, QInputDialog
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import analysis_algorithm_dict
from PartSegCore.analysis.load_functions import LoadProfileFromJSON
from PartSegCore.analysis.measurement_calculation import Volume, Voxels
from PartSegCore.analysis.save_functions import SaveProfilesToJSON
from PartSegCore.mask.algorithm_description import mask_algorithm_dict
from PartSegCore.segmentation import ROIExtractionResult

napari_skip = pytest.mark.skipif(
    packaging.version.parse(napari.__version__) < packaging.version.parse("0.4.10"), reason="To old napari"
)

napari_4_11_skip = pytest.mark.skipif(
    packaging.version.parse(napari.__version__) == packaging.version.parse("0.4.11"), reason="To old napari"
)


@pytest.fixture(autouse=True)
def clean_settings(tmp_path):
    old_settings = _settings._settings
    _settings._settings = None
    # _settings._settings = BaseSettings(tmp_path)
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

    res = ROIExtractionResult(
        np.ones((10, 10), dtype=np.uint8), widget.profile_dict["prof3"], roi_annotation={1: {"foo": 1, "baz": 2}}
    )
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
    monkeypatch.setattr(CustomSaveDialog, "exec_", lambda x: True)
    monkeypatch.setattr(
        CustomSaveDialog,
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
def test_simple_measurement_create(make_napari_viewer, qtbot):
    from PartSeg.plugins.napari_widgets.simple_measurement_widget import SimpleMeasurement

    data = np.zeros((10, 10), dtype=np.uint8)

    viewer = make_napari_viewer()
    viewer.add_labels(data, name="label")
    measurement = SimpleMeasurement(viewer)
    viewer.window.add_dock_widget(measurement)
    measurement.reset_choices()
    measurement.labels_choice.reset_choices()
    measurement.labels_choice.value = viewer.layers["label"]
    for el in measurement.measurement_layout:
        if el.text in [Volume.get_name(), Voxels.get_name()]:
            el.value = True

    measurement._calculate()
    for _ in range(10):
        qtbot.wait(200)
        if measurement.calculate_btn.enabled:
            break

    assert measurement.calculate_btn.enabled


@napari_skip
@pytest.mark.enablethread
@pytest.mark.enabledialog
def test_measurement_create(make_napari_viewer, qtbot, bundle_test_dir):
    from PartSeg.plugins.napari_widgets.measurement_widget import Measurement

    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 2:-2] = 1
    data[5:-2, 2:-2] = 2

    viewer = make_napari_viewer()
    viewer.add_labels(data, name="label")
    viewer.add_image(data, name="image")
    measurement = Measurement(viewer)
    viewer.window.add_dock_widget(measurement)
    measurement.reset_choices()
    measurement_data = measurement.settings.load_metadata(str(bundle_test_dir / "napari_measurements_profile.json"))
    measurement.settings.measurement_profiles["test"] = measurement_data["test"]
    assert measurement.measurement_widget.measurement_type.count() == 2
    measurement.measurement_widget.measurement_type.setCurrentIndex(1)
    assert measurement.measurement_widget.measurement_type.currentText() == "test"
    assert measurement.measurement_widget.recalculate_button.isEnabled()
    assert measurement.measurement_widget.check_if_measurement_can_be_calculated("test") == "test"
    measurement.measurement_widget.append_measurement_result()


def test_mask_create(make_napari_viewer, qtbot):
    data = np.zeros((10, 10), dtype=np.uint8)

    viewer = make_napari_viewer()
    viewer.add_labels(data, name="label")
    mask_create = MaskCreateNapari(viewer)
    viewer.window.add_dock_widget(mask_create)
    mask_create.reset_choices()
    mask_create.create_mask()
    assert mask_create.mask_widget.get_dilate_radius() == 0

    assert "Mask" in viewer.layers


@napari_4_11_skip
@pytest.mark.enablethread
def test_search_labels(make_napari_viewer, qtbot):
    viewer = make_napari_viewer()
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 2:-2] = 1
    data[5:-2, 2:-2] = 2
    viewer.add_labels(data, name="label")
    search = SearchLabel(napari_viewer=viewer)
    viewer.window.add_dock_widget(search)

    search.search_type.value = SearchType.Highlight
    search.component_selector.value = 1
    assert ".Highlight" in viewer.layers
    qtbot.wait(500)
    assert ".Highlight" in viewer.layers
    search._stop()
    assert ".Highlight" not in viewer.layers
    search.search_type.value = SearchType.Zoom_in
    search.search_type.value = SearchType.Highlight
    assert ".Highlight" in viewer.layers
    search.search_type.value = SearchType.Zoom_in
    assert ".Highlight" not in viewer.layers
    search.search_type.value = SearchType.Highlight
    search.component_selector.value = 2
