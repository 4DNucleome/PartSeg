from contextlib import suppress

import napari
import packaging.version
import pytest
from qtpy.QtWidgets import QApplication, QDialog, QMessageBox

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_mask.main_window import ChosenComponents
from PartSeg._roi_mask.stack_settings import StackSettings
from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui import napari_image_view
from PartSegCore.algorithm_describe_base import ROIExtractionProfile
from PartSegCore.analysis import SegmentationPipeline, SegmentationPipelineElement
from PartSegCore.segmentation.restartable_segmentation_algorithms import BorderRim, LowerThresholdAlgorithm


@pytest.fixture
def base_settings(image, tmp_path, measurement_profiles):
    settings = BaseSettings(tmp_path)
    settings.image = image
    return settings


@pytest.fixture
def part_settings(image, tmp_path, measurement_profiles):
    settings = PartSettings(tmp_path)
    settings.image = image
    for el in measurement_profiles:
        settings.measurement_profiles[el.name] = el
    return settings


@pytest.fixture
def stack_settings(tmp_path):
    settings = StackSettings(tmp_path)
    chose = ChosenComponents()
    settings.chosen_components_widget = chose
    yield settings
    chose.deleteLater()


@pytest.fixture
def part_settings_with_project(image, analysis_segmentation2, tmp_path):
    settings = PartSettings(tmp_path)
    settings.image = image
    settings.set_project_info(analysis_segmentation2)
    return settings


@pytest.fixture
def border_rim_profile():
    return ROIExtractionProfile("border_profile", BorderRim.get_name(), BorderRim.get_default_values())


@pytest.fixture
def lower_threshold_profile():
    return ROIExtractionProfile(
        "lower_profile", LowerThresholdAlgorithm.get_name(), LowerThresholdAlgorithm.get_default_values()
    )


@pytest.fixture
def sample_pipeline(border_rim_profile, lower_threshold_profile, mask_property):
    return SegmentationPipeline(
        "sample_pipeline", border_rim_profile, [SegmentationPipelineElement(lower_threshold_profile, mask_property)]
    )


@pytest.fixture(autouse=True)
def disable_threads_viewer(monkeypatch):
    def _prepare_layers(self, image, parameters, replace):
        self._add_image(napari_image_view._prepare_layers(image, parameters, replace))

    monkeypatch.setattr(napari_image_view.ImageView, "_prepare_layers", _prepare_layers)

    def _add_layer_util(self, index, layer, filters):
        self.viewer.add_layer(layer)

    monkeypatch.setattr(napari_image_view.ImageView, "_add_layer_util", _add_layer_util)


@pytest.fixture(autouse=True)
def check_opened_windows(qapp):
    yield
    widgets = qapp.topLevelWidgets()
    for widget in widgets:
        assert not widget.isVisible()


@pytest.fixture(autouse=True)
def block_threads(monkeypatch, request):
    if "enablethread" in request.keywords:
        return

    from pytestqt.qt_compat import qt_api
    from qtpy.QtCore import QThread, QTimer

    old_start = QTimer.start

    class OldTimer(QTimer):
        def start(self, time=None):
            if time is not None:
                old_start(self, time)
            else:
                old_start(self)

    def not_start(self):
        raise RuntimeError("Thread should not be used in test")

    monkeypatch.setattr(QTimer, "start", not_start)
    monkeypatch.setattr(QThread, "start", not_start)
    monkeypatch.setattr(qt_api.QtCore, "QTimer", OldTimer)


@pytest.fixture(autouse=True)
def clean_settings():
    try:
        try:
            from napari.settings import SETTINGS
        except ImportError:
            from napari.utils.settings import SETTINGS

        SETTINGS.reset()
        yield
        with suppress(AttributeError):
            SETTINGS.reset()
    except ImportError:
        yield


@pytest.fixture
def leaked_widgets():
    initial = QApplication.topLevelWidgets()
    yield
    QApplication.processEvents()
    leak = set(QApplication.topLevelWidgets()).difference(initial)
    if any(n.__class__.__name__ != "CanvasBackendDesktop" for n in leak):
        raise AssertionError(f"Widgets ({len(leak)}) leaked!: {leak}")


@pytest.fixture(autouse=True)
def reset_napari_settings(monkeypatch, tmp_path):
    napari_version = packaging.version.parse(napari.__version__)

    def _mock_save(self, path=None, **dict_kwargs):
        return  # skipcq: PTC-W0049

    if napari_version < packaging.version.parse("0.4.10"):
        from napari.utils.settings import _manager

        monkeypatch.setattr(_manager.SettingsManager, "_save", _mock_save)

        _manager.SETTINGS = _manager.SettingsManager(tmp_path)

    elif napari_version == packaging.version.parse("0.4.10"):
        from napari.utils.settings import _manager

        monkeypatch.setattr(_manager, "user_config_dir", lambda *x: tmp_path)
        monkeypatch.setattr(_manager.SettingsManager, "_save", _mock_save)

        _manager.SETTINGS = _manager._SettingsProxy()

    else:
        from napari import settings

        cp = settings.NapariSettings.__private_attributes__["_config_path"]
        monkeypatch.setattr(cp, "default", tmp_path / "save.yaml")
        monkeypatch.setattr(settings.NapariSettings, "save", _mock_save)
        settings._SETTINGS = None

    yield


@pytest.fixture(autouse=True)
def block_message_box(monkeypatch, request):
    def raise_on_call(*_, **__):
        raise RuntimeError("exec_ call")

    monkeypatch.setattr(QMessageBox, "exec_", raise_on_call)
    monkeypatch.setattr(QMessageBox, "critical", raise_on_call)
    monkeypatch.setattr(QMessageBox, "information", raise_on_call)
    monkeypatch.setattr(QMessageBox, "question", raise_on_call)
    monkeypatch.setattr(QMessageBox, "warning", raise_on_call)
    if "enabledialog" not in request.keywords:
        monkeypatch.setattr(QDialog, "exec_", raise_on_call)
