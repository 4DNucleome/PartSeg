import contextlib
from contextlib import suppress

import pytest
from qtpy.QtWidgets import QDialog, QInputDialog, QMessageBox

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg._roi_mask.main_window import ChosenComponents
from PartSeg._roi_mask.stack_settings import StackSettings
from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui import napari_image_view


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
def stack_settings(tmp_path, image):
    settings = StackSettings(tmp_path)
    settings.image = image
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


@pytest.fixture(autouse=True)
def _disable_threads_viewer_patch_prepare_leyers(monkeypatch):
    def _prepare_layers(self, image, parameters, replace):
        self._add_image(napari_image_view._prepare_layers(image, parameters, replace))

    monkeypatch.setattr(napari_image_view.ImageView, "_prepare_layers", _prepare_layers)


@pytest.fixture(autouse=True)
def _disable_threads_viewer_patch_add_layer(monkeypatch, request):
    if "no_patch_add_layer" in request.keywords:
        return

    def _add_layer_util(self, index, layer, filters):
        if layer not in self.viewer.layers:
            self.viewer.add_layer(layer)

        monkeypatch.setattr(napari_image_view.ImageView, "_add_layer_util", _add_layer_util)


@pytest.fixture(autouse=True)
def _check_opened_windows(qapp):
    yield
    widgets = qapp.topLevelWidgets()
    for widget in widgets:
        assert not widget.isVisible()


@pytest.fixture(autouse=True)
def _block_threads(monkeypatch, request):
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
def _clean_settings():
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


@pytest.fixture(autouse=True)
def _reset_napari_settings(monkeypatch, tmp_path):
    def _mock_save(self, path=None, **dict_kwargs):
        return  # skipcq: PTC-W0049

    from napari import settings

    cp = settings.NapariSettings.__private_attributes__["_config_path"]
    monkeypatch.setattr(cp, "default", tmp_path / "save.yaml")
    monkeypatch.setattr(settings.NapariSettings, "save", _mock_save)
    settings._SETTINGS = None


@pytest.fixture(autouse=True)
def _block_message_box(monkeypatch, request):
    def raise_on_call(*_, **__):
        raise RuntimeError("exec_ call")  # pragma: no cover

    monkeypatch.setattr(QMessageBox, "exec_", raise_on_call)
    monkeypatch.setattr(QMessageBox, "critical", raise_on_call)
    monkeypatch.setattr(QMessageBox, "information", raise_on_call)
    monkeypatch.setattr(QMessageBox, "question", raise_on_call)
    monkeypatch.setattr(QMessageBox, "warning", raise_on_call)
    monkeypatch.setattr("PartSeg.common_gui.error_report.QMessageFromException.exec_", raise_on_call)
    monkeypatch.setattr(QInputDialog, "getText", raise_on_call)
    if "enabledialog" not in request.keywords:
        monkeypatch.setattr(QDialog, "exec_", raise_on_call)


class DummyConnect:
    def __init__(self, li):
        self.li = li

    def connect(self, func):
        self.li.append(func)


class DummyThrottler:
    def __init__(self, *args, **kwargs):
        self._call_list = []

    def setTimeout(self, *args, **kwargs):
        pass  # as it is dummy throttler then timeout is obsolete.

    def throttle(self, *args, **kwargs):
        for cl in self._call_list:
            cl(*args, **kwargs)

    @property
    def triggered(self):
        return DummyConnect(self._call_list)


@pytest.fixture(autouse=True)
def _mock_throttler(monkeypatch):
    with contextlib.suppress(ImportError):
        from napari._qt import qt_main_window

        if hasattr(qt_main_window, "QSignalThrottler"):
            monkeypatch.setattr(qt_main_window, "QSignalThrottler", DummyThrottler)
    with contextlib.suppress(ImportError):
        monkeypatch.setattr(
            "napari._qt.threads.status_checker.StatusChecker.start",
            lambda x: None,
            raising=False,
        )
