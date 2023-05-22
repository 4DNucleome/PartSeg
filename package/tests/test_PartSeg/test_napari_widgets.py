import contextlib
import gc
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from napari.layers import Image as NapariImage
from napari.layers import Labels
from napari.utils import Colormap
from qtpy.QtCore import QTimer

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_analysis.profile_export import ExportDialog, ImportDialog
from PartSeg.common_gui.custom_load_dialog import CustomLoadDialog
from PartSeg.common_gui.custom_save_dialog import CustomSaveDialog
from PartSeg.common_gui.napari_image_view import SearchType
from PartSeg.plugins.napari_widgets import (
    ImageColormap,
    MaskCreate,
    ROIAnalysisExtraction,
    ROIMaskExtraction,
    SearchLabel,
    _settings,
)
from PartSeg.plugins.napari_widgets.algorithm_widgets import (
    BorderSmoothingModel,
    DoubleThresholdModel,
    NoiseFilterModel,
    Threshold,
    ThresholdModel,
)
from PartSeg.plugins.napari_widgets.colormap_control import NapariColormapControl
from PartSeg.plugins.napari_widgets.lables_control import LabelSelector, NapariLabelShow
from PartSeg.plugins.napari_widgets.measurement_widget import update_properties
from PartSeg.plugins.napari_widgets.roi_extraction_algorithms import ProfilePreviewDialog, QInputDialog
from PartSeg.plugins.napari_widgets.search_label_widget import HIGHLIGHT_LABEL_NAME
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis.algorithm_description import AnalysisAlgorithmSelection
from PartSegCore.analysis.load_functions import LoadProfileFromJSON
from PartSegCore.analysis.measurement_calculation import Volume, Voxels
from PartSegCore.analysis.save_functions import SaveProfilesToJSON
from PartSegCore.mask.algorithm_description import MaskAlgorithmSelection
from PartSegCore.segmentation import ROIExtractionResult
from PartSegCore.segmentation.border_smoothing import SmoothAlgorithmSelection
from PartSegCore.segmentation.noise_filtering import NoiseFilterSelection
from PartSegCore.segmentation.threshold import DoubleThresholdSelection, ThresholdSelection


@pytest.fixture(autouse=True)
def _clean_settings(tmp_path):
    old_settings = _settings._SETTINGS
    _settings._SETTINGS = PartSettings(tmp_path)
    yield
    _settings._SETTINGS = old_settings


def no_action(*_):  # skipcq: PTC-W0049
    pass


def test_dummy_for_gc():
    gc.collect()


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
    widget.mask_name = "1"
    widget.update_image()
    assert not widget.mask_name
    # check if mask is not cleaned if channels does not change
    widget.mask_name = "1"
    widget.update_image()
    assert widget.mask_name == "1"

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


@pytest.mark.parametrize("register", [AnalysisAlgorithmSelection, MaskAlgorithmSelection])
def test_profile_preview_dialog(part_settings, register, qtbot, monkeypatch, tmp_path):
    alg = register.get_default()
    profiles = {
        "prof1": ROIExtractionProfile(name="prof1", algorithm=alg.name, values=alg.values),
        "prof2": ROIExtractionProfile(name="prof2", algorithm=alg.name, values=alg.values),
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
    qtbot.wait_until(lambda: measurement.calculate_btn.enabled)

    assert measurement.calculate_btn.enabled


@pytest.mark.enablethread()
@pytest.mark.enabledialog()
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


def test_update_properties():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 2:-2] = 1
    data[5:-2, 2:-2] = 2
    labels = Labels(data)
    df = pd.DataFrame([[0, 0], [1, 1]], columns=["x", "y"])
    df2 = pd.DataFrame([[0, 0], [1, 1]], columns=["x", "z"])
    update_properties(df, labels, True)
    assert len(labels.properties) == 2
    assert "x" in labels.properties
    assert "y" in labels.properties
    assert np.all(labels.properties["x"] == np.array([0, 1]))
    update_properties(df2, labels, False)
    assert len(labels.properties) == 3
    assert "z" in labels.properties
    update_properties(df2, labels, True)
    assert len(labels.properties) == 2
    assert "y" not in labels.properties


def test_mask_create(make_napari_viewer, qtbot):
    data = np.zeros((10, 10), dtype=np.uint8)

    viewer = make_napari_viewer()
    viewer.add_labels(data, name="label")
    mask_create = MaskCreate(viewer)
    viewer.window.add_dock_widget(mask_create)
    mask_create.reset_choices()
    mask_create.create_mask()
    assert mask_create.mask_widget.get_dilate_radius() == 0

    assert "Mask" in viewer.layers


@pytest.fixture()
def _shutdown_timers(monkeypatch):
    register = []
    old_start = QTimer.start

    def mock_start(self, interval=None):
        register.append(self)
        return old_start(self) if interval is None else old_start(self, interval)

    monkeypatch.setattr(QTimer, "start", mock_start)
    yield

    for timer in register:
        with contextlib.suppress(RuntimeError):
            timer.stop()

    for timer in register:
        with contextlib.suppress(RuntimeError):
            assert not timer.isActive()


@pytest.mark.enablethread()
@pytest.mark.usefixtures("_shutdown_timers")
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
    assert HIGHLIGHT_LABEL_NAME in viewer.layers
    qtbot.wait(500)
    assert HIGHLIGHT_LABEL_NAME in viewer.layers
    search._stop()
    assert HIGHLIGHT_LABEL_NAME not in viewer.layers
    search.search_type.value = SearchType.Zoom_in
    search.search_type.value = SearchType.Highlight
    assert HIGHLIGHT_LABEL_NAME in viewer.layers
    search.search_type.value = SearchType.Zoom_in
    assert HIGHLIGHT_LABEL_NAME not in viewer.layers
    search.search_type.value = SearchType.Highlight
    search.component_selector.value = 2


@pytest.fixture()
def viewer_with_data(make_napari_viewer):
    viewer = make_napari_viewer()
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 2:-2] = 1
    data[5:-2, 2:-2] = 2
    viewer.add_image(data, name="image")
    viewer.add_labels(data, name="label")
    viewer.layers.selection.clear()
    return viewer


def test_label_control(viewer_with_data, qtbot):
    widget = LabelSelector(viewer_with_data)
    qtbot.addWidget(widget)
    widget.label_view.refresh()

    assert widget.label_view.layout().count() == 2  # labels and stretch

    viewer_with_data.layers.selection.add(viewer_with_data.layers["label"])
    assert widget.label_view.layout().itemAt(0).widget().apply_label_btn.isEnabled()
    viewer_with_data.layers.selection.clear()
    viewer_with_data.layers.selection.add(viewer_with_data.layers["image"])
    assert not widget.label_view.layout().itemAt(0).widget().apply_label_btn.isEnabled()


def test_image_colormap(viewer_with_data, qtbot):
    widget = ImageColormap(viewer_with_data)
    qtbot.addWidget(widget)
    widget.colormap_list.refresh()

    viewer_with_data.layers.selection.add(viewer_with_data.layers["label"])
    assert not widget.colormap_list.grid_layout.itemAt(0).widget().apply_colormap_btn.isEnabled()
    viewer_with_data.layers.selection.clear()
    viewer_with_data.layers.selection.add(viewer_with_data.layers["image"])
    assert widget.colormap_list.grid_layout.itemAt(0).widget().apply_colormap_btn.isEnabled()


def test_napari_label_show(viewer_with_data, qtbot):
    widget = NapariLabelShow(viewer_with_data, "label_name", [[128, 234, 123], [240, 0, 0]], False)
    qtbot.addWidget(widget)
    viewer_with_data.layers.selection.add(viewer_with_data.layers["image"])
    assert not widget.apply_label_btn.isEnabled()
    viewer_with_data.layers.selection.add(viewer_with_data.layers["label"])
    assert not widget.apply_label_btn.isEnabled()
    viewer_with_data.layers.selection.remove(viewer_with_data.layers["image"])
    assert widget.apply_label_btn.isEnabled()
    assert viewer_with_data.layers["label"].color_mode == "auto"
    widget.apply_label_btn.click()
    assert viewer_with_data.layers["label"].color_mode == "direct"


def test_napari_colormap_control(viewer_with_data, qtbot):
    widget = NapariColormapControl(viewer_with_data, Colormap([[0, 0, 0, 0], [1, 0, 0, 1]]), False, "image_name")
    qtbot.addWidget(widget)
    viewer_with_data.layers.selection.add(viewer_with_data.layers["label"])
    assert not widget.apply_colormap_btn.isEnabled()
    viewer_with_data.layers.selection.add(viewer_with_data.layers["image"])
    assert not widget.apply_colormap_btn.isEnabled()
    viewer_with_data.layers.selection.remove(viewer_with_data.layers["label"])
    assert widget.apply_colormap_btn.isEnabled()
    assert viewer_with_data.layers["image"].colormap.name == "gray"
    widget.apply_colormap_btn.click()
    assert viewer_with_data.layers["image"].colormap.name == "custom"


@pytest.fixture(params=range(2, 5))
def ndim_param(request):
    return request.param


@pytest.fixture()
def napari_image(ndim_param):
    shape = (1,) * max(ndim_param - 3, 0) + (10,) * min(ndim_param, 3)
    slice_ = (slice(None),) * max(ndim_param - 3, 0) + (slice(2, 8),) * min(ndim_param, 3)
    data = np.zeros(shape, dtype=np.uint16)
    data[slice_] = 10000
    data[(0,) * (ndim_param - 2) + (slice(4, 6), slice(4, 6))] = 20000
    return NapariImage(data)


@pytest.fixture()
def napari_labels(ndim_param):
    shape = (1,) * max(ndim_param - 3, 0) + (10,) * min(ndim_param, 3)
    slice_ = (slice(None),) * max(ndim_param - 3, 0) + (slice(2, 8),) * min(ndim_param, 3)
    data = np.zeros(shape, dtype=np.uint8)
    data[slice_] = 1
    data[..., -3, -3] = 0
    return Labels(data)


@pytest.mark.parametrize("algorithm", ThresholdSelection.__register__.values())
def test_threshold_model(algorithm, napari_image):
    if algorithm.get_name() == "Kittler Illingworth":
        pytest.skip("Kittler Illingworth is not working properly")
    alg_params = ThresholdSelection(
        name=algorithm.get_name(),
        values=algorithm.__argument_class__(),
    )
    model = ThresholdModel(threshold=alg_params, data=napari_image)
    assert model.run_calculation()["layer_type"] == "labels"


@pytest.mark.parametrize("algorithm", DoubleThresholdSelection.__register__.values())
def test_double_threshold_model(algorithm, napari_image):
    alg_params = DoubleThresholdSelection(
        name=algorithm.get_name(),
        values=algorithm.__argument_class__(),
    )
    model = DoubleThresholdModel(threshold=alg_params, data=napari_image)
    assert model.run_calculation()["layer_type"] == "labels"


@pytest.mark.parametrize("algorithm", NoiseFilterSelection.__register__.values())
def test_noise_filter_model(algorithm, napari_image):
    alg_params = NoiseFilterSelection(
        name=algorithm.get_name(),
        values=algorithm.__argument_class__(),
    )
    model = NoiseFilterModel(noise_filtering=alg_params, data=napari_image)
    assert model.run_calculation()["layer_type"] == "image"


@pytest.mark.parametrize("algorithm", SmoothAlgorithmSelection.__register__.values())
def test_border_smoothing_model(algorithm, napari_labels):
    alg_params = SmoothAlgorithmSelection(
        name=algorithm.get_name(),
        values=algorithm.__argument_class__(),
    )
    model = BorderSmoothingModel(border_smoothing=alg_params, data=napari_labels)
    assert model.run_calculation()["layer_type"] == "labels"


@patch("PartSeg.plugins.napari_widgets.algorithm_widgets.show_info")
def test_threshold_widget(show_patch, make_napari_viewer, qtbot, napari_image):
    viewer = make_napari_viewer()
    viewer.add_layer(napari_image)
    widget = Threshold(viewer)
    viewer.window.add_dock_widget(widget, area="right")
    widget.reset_choices()
    assert len(viewer.layers) == 1
    widget.run_operation()
    assert len(viewer.layers) == 2
    assert "Labels" in viewer.layers
    widget.run_operation()
    assert len(viewer.layers) == 2
    del viewer.layers[napari_image.name]
    assert len(viewer.layers) == 1
    widget.run_operation()
    assert len(viewer.layers) == 1
    assert show_patch.called
    new_image = NapariImage(
        np.reshape(napari_image.data, (1, *napari_image.data.shape)), scale=(1, *napari_image.scale)
    )
    viewer.add_layer(new_image)
    widget.run_operation()
    assert len(viewer.layers) == 3
