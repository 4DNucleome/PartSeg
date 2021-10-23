import argparse
import sys
import typing
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pytest
import sentry_sdk
from packaging.version import parse
from qtpy.QtWidgets import QMessageBox

from PartSeg.common_backend import (
    base_argparser,
    base_settings,
    except_hook,
    load_backup,
    partially_const_dict,
    progress_thread,
    segmentation_thread,
)
from PartSeg.common_gui.error_report import ErrorDialog
from PartSegCore import state_store
from PartSegCore.algorithm_describe_base import AlgorithmProperty, ROIExtractionProfile
from PartSegCore.roi_info import ROIInfo
from PartSegCore.segmentation import ROIExtractionResult
from PartSegCore.segmentation.algorithm_base import ROIExtractionAlgorithm, SegmentationLimitException
from PartSegImage import Image, TiffFileException

IS_MACOS = sys.platform == "darwin"


class TestExceptHook:
    def test_show_error(self, monkeypatch, qtbot):
        exec_list = []

        def exec_mock(self):
            qtbot.add_widget(self)
            exec_list.append(self)

        monkeypatch.setattr(except_hook.QMessageBox, "exec_", exec_mock)
        monkeypatch.setattr(ErrorDialog, "exec_", exec_mock)

        except_hook.show_error()
        assert exec_list == []
        except_hook.show_error(TiffFileException("sample"))
        assert len(exec_list) == 1, "exec not called"
        message = exec_list[0]
        assert isinstance(message, QMessageBox)
        assert message.icon() == QMessageBox.Critical
        assert message.windowTitle() == "Tiff error" or IS_MACOS
        assert message.text().startswith("During read file there is an error")

        exec_list = []
        except_hook.show_error(SegmentationLimitException("sample"))
        assert len(exec_list) == 1, "exec not called"
        message = exec_list[0]
        assert isinstance(message, QMessageBox)
        assert message.icon() == QMessageBox.Critical
        assert message.windowTitle() == "Segmentation limitations" or IS_MACOS
        assert message.text().startswith("During segmentation process algorithm meet limitations")

        exec_list = []
        try:
            raise RuntimeError("aaa")
        except RuntimeError as e:
            except_hook.show_error(e)
        assert len(exec_list) == 1, "exec not called"
        dialog = exec_list[0]
        assert isinstance(dialog, ErrorDialog)

    @pytest.mark.parametrize("header", [None, "Text"])
    @pytest.mark.parametrize("text", [None, "Long text"])
    def test_show_warning(self, monkeypatch, header, text, qtbot):
        exec_list = []

        def exec_mock(self):
            qtbot.add_widget(self)
            exec_list.append(self)

        monkeypatch.setattr(except_hook.QMessageBox, "exec_", exec_mock)
        except_hook.show_warning(header, text)
        assert len(exec_list) == 1

    def test_my_excepthook(self, monkeypatch, capsys):
        catch_list = []
        error_list = []
        exit_list = []
        sentry_catch_list = []

        def excepthook_catch(type_, value, trace_back):
            catch_list.append((type_, value, trace_back))

        def exit_catch(value):
            exit_list.append(value)

        def capture_exception_catch(value):
            sentry_catch_list.append(value)

        def show_error_catch(value):
            error_list.append(value)

        def import_raise(_value):
            raise ImportError()

        monkeypatch.setattr(sys, "__excepthook__", excepthook_catch)
        monkeypatch.setattr(sys, "exit", exit_catch)
        monkeypatch.setattr(sentry_sdk, "capture_exception", capture_exception_catch)
        monkeypatch.setattr(except_hook, "show_error", show_error_catch)
        monkeypatch.setattr(except_hook, "parsed_version", parse("0.13.12"))

        monkeypatch.setattr(state_store, "show_error_dialog", False)
        except_hook.my_excepthook(KeyboardInterrupt, KeyboardInterrupt(), [])
        assert exit_list == [1]
        assert capsys.readouterr().err == "KeyboardInterrupt close\n"

        except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
        assert len(catch_list) == 1
        assert catch_list[0][0] is RuntimeError
        catch_list = []

        monkeypatch.setattr(state_store, "show_error_dialog", True)

        except_hook.my_excepthook(KeyboardInterrupt, KeyboardInterrupt(), [])
        assert exit_list == [1, 1]
        assert capsys.readouterr().err == "KeyboardInterrupt close\n"

        except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
        assert len(error_list) == 1
        assert isinstance(error_list[0], RuntimeError)
        assert len(sentry_catch_list) == 0

        monkeypatch.setattr(except_hook, "show_error", import_raise)

        monkeypatch.setattr(state_store, "report_errors", True)
        monkeypatch.setattr(except_hook, "parsed_version", parse("0.13.12dev1"))
        except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
        assert len(sentry_catch_list) == 1
        assert isinstance(sentry_catch_list[0], RuntimeError)
        assert len(catch_list) == 1
        assert catch_list[0][0] is RuntimeError


class TestBaseArgparse:
    def test_proper_suffix(self):
        assert base_argparser.proper_suffix("aaa") == "aaa"
        assert base_argparser.proper_suffix("123") == "123"
        assert base_argparser.proper_suffix("123aa") == "123aa"
        with pytest.raises(argparse.ArgumentTypeError):
            base_argparser.proper_suffix("123aa#")

    def test_proper_path(self, tmp_path, monkeypatch):
        def raise_os_error(_path):
            raise OSError("a")

        assert base_argparser.proper_path(str(tmp_path)) == str(tmp_path)
        assert base_argparser.proper_path(str(tmp_path / "aaa")) == str(tmp_path / "aaa")

        monkeypatch.setattr(base_argparser.os, "makedirs", raise_os_error)
        with pytest.raises(argparse.ArgumentTypeError):
            base_argparser.proper_path(str(tmp_path / "dddddd"))

    def test_custom_parser(self, monkeypatch):
        state_store_mock = argparse.Namespace(save_folder="/")
        monkeypatch.setattr(base_argparser, "state_store", state_store_mock)
        monkeypatch.setattr(base_argparser, "state_store", state_store_mock)
        monkeypatch.setattr(base_argparser, "_setup_sentry", lambda: 1)
        monkeypatch.setattr(base_argparser, "my_excepthook", sys.excepthook)
        monkeypatch.setattr(base_argparser.locale, "setlocale", lambda x, y: 1)

        parser = base_argparser.CustomParser()
        parser.parse_args([])

    def test_safe_repr(self):
        assert base_argparser.safe_repr(1) == "1"
        assert base_argparser.safe_repr(np.arange(3)) == "array([0, 1, 2])"


class TestProgressThread:
    def test_progress_thread(self, qtbot):
        thr = progress_thread.ProgressTread()
        with qtbot.waitSignal(thr.range_changed, check_params_cb=lambda x, y: x == 0 and y == 15):
            thr.info_function("max", 15)
        with qtbot.waitSignal(thr.step_changed, check_params_cb=lambda x: x == 10):
            thr.info_function("step", 10)

    def test_execute_function_thread_no_callback(self, qtbot):
        def no_callback_fun(a, b):
            return a + b

        thr = progress_thread.ExecuteFunctionThread(no_callback_fun, [1], {"b": 4})
        thr.run()
        assert thr.result == 5

    def test_execute_function_thread_callback_fun(self, qtbot):
        def callback_fun(a, b, callback_function):
            callback_function("max", 15)
            return a + b

        thr = progress_thread.ExecuteFunctionThread(callback_fun, [1], {"b": 4})
        with qtbot.waitSignal(thr.range_changed, check_params_cb=lambda x, y: x == 0 and y == 15):
            thr.run()
        assert thr.result == 5

    def test_execute_function_thread_separate_callback(self, qtbot):
        def callback_fun2(a, b, range_changed, step_changed):
            range_changed(1, 15)
            step_changed(10)
            return a + b

        thr = progress_thread.ExecuteFunctionThread(callback_fun2, [1], {"b": 4})
        with qtbot.waitSignals(
            [thr.range_changed, thr.step_changed], check_params_cbs=[lambda x, y: x == 1 and y == 15, lambda x: x == 10]
        ):
            thr.run()
        assert thr.result == 5

    def test_execute_function_thread_exception(self, qtbot):
        def callback_fun2(a, b):
            raise RuntimeError(a + b)

        thr = progress_thread.ExecuteFunctionThread(callback_fun2, [1], {"b": 4})
        with qtbot.waitSignal(thr.error_signal, check_params_cb=lambda x: x.args[0] == 5):
            thr.run()


class TestSegmentationThread:
    def test_empty_image(self, capsys):
        thr = segmentation_thread.SegmentationThread(ROIExtractionAlgorithmForTest())
        assert thr.get_info_text() == "text"
        thr.set_parameters(a=1)
        thr.run()
        assert capsys.readouterr().err.startswith("No image in")

    def test_run(self, qtbot):
        algorithm = ROIExtractionAlgorithmForTest()
        thr = segmentation_thread.SegmentationThread(algorithm)
        image = Image(np.zeros((10, 10), dtype=np.uint8), image_spacing=(1, 1), axes_order="XY")
        algorithm.set_image(image)
        with qtbot.waitSignals([thr.execution_done, thr.progress_signal]):
            thr.run()

    def test_run_return_none(self, qtbot):
        algorithm = ROIExtractionAlgorithmForTest(return_none=True)
        thr = segmentation_thread.SegmentationThread(algorithm)
        image = Image(np.zeros((10, 10), dtype=np.uint8), image_spacing=(1, 1), axes_order="XY")
        algorithm.set_image(image)
        with qtbot.assertNotEmitted(thr.execution_done):
            thr.run()

    def test_run_exception(self, qtbot):
        algorithm = ROIExtractionAlgorithmForTest(raise_=True)
        thr = segmentation_thread.SegmentationThread(algorithm)
        image = Image(np.zeros((10, 10), dtype=np.uint8), image_spacing=(1, 1), axes_order="XY")
        algorithm.set_image(image)
        with qtbot.assertNotEmitted(thr.execution_done):
            with qtbot.waitSignal(thr.exception_occurred):

                thr.run()

    def test_running_set_parameters(self, qtbot, monkeypatch):
        thr = segmentation_thread.SegmentationThread(ROIExtractionAlgorithmForTest())
        thr.set_parameters(a=1)
        monkeypatch.setattr(thr, "isRunning", lambda: True)
        thr.set_parameters(a=2)
        assert thr.algorithm.new_parameters["a"] == 1
        thr.finished_task()
        assert thr.algorithm.new_parameters["a"] == 2

    def test_running_clean(self, qtbot, monkeypatch):
        clean_list = []
        thr = segmentation_thread.SegmentationThread(ROIExtractionAlgorithmForTest())
        monkeypatch.setattr(thr.algorithm, "clean", lambda: clean_list.append(1))
        thr.clean()
        assert clean_list == [1]
        monkeypatch.setattr(thr, "isRunning", lambda: True)
        thr.clean()
        assert clean_list == [1]
        thr.finished_task()
        assert clean_list == [1, 1]

    def test_running_start(self, qtbot, monkeypatch):
        start_list = []
        monkeypatch.setattr(segmentation_thread.QThread, "start", lambda x, y: start_list.append(1))
        thr = segmentation_thread.SegmentationThread(ROIExtractionAlgorithmForTest())
        thr.start()
        assert start_list == [1]
        monkeypatch.setattr(thr, "isRunning", lambda: True)
        thr.start()
        assert start_list == [1]
        thr.finished_task()
        assert start_list == [1, 1]


class ROIExtractionAlgorithmForTest(ROIExtractionAlgorithm):
    def __init__(self, raise_=False, return_none=False):
        super().__init__()
        self.raise_ = raise_
        self.return_none = return_none

    @classmethod
    def support_time(cls):
        return False

    @classmethod
    def support_z(cls):
        return True

    def calculation_run(self, report_fun: Callable[[str, int], None]) -> Optional[ROIExtractionResult]:
        if self.raise_:
            raise RuntimeError("ee")
        if self.return_none:
            return
        report_fun("text", 1)
        return ROIExtractionResult(np.zeros((10, 10), dtype=np.uint8), ROIExtractionProfile("a", "a", {}))

    def get_info_text(self):
        return "text"

    @classmethod
    def get_name(cls) -> str:
        return "test"

    @classmethod
    def get_fields(cls) -> typing.List[typing.Union[AlgorithmProperty, str]]:
        return [AlgorithmProperty("a", "A", 0)]


class TestPartiallyConstDict:
    def test_add_remove(self, qtbot):
        class TestDict(partially_const_dict.PartiallyConstDict):
            const_item_dict = {"custom_a": 1, "b": 2}

        data = {"c": 3, "d": 4}
        dkt = TestDict(data)
        assert set(dkt) == {"custom_a", "b", "c", "d"}
        assert len(dkt) == 4
        with pytest.raises(ValueError):
            dkt["b"] = 1
        with pytest.raises(ValueError):
            dkt["custom_a"] = 1
        with qtbot.waitSignal(dkt.item_added):
            dkt["e"] = 7
        assert len(dkt) == 5
        assert dkt["b"] == (2, False)
        assert dkt["e"] == (7, True)
        with pytest.raises(KeyError):
            dkt["k"]

        with pytest.raises(ValueError):
            del dkt["b"]
        with qtbot.waitSignal(dkt.item_removed):
            del dkt["e"]
        assert len(dkt) == 4
        assert dkt.get_position("b") == 1
        with pytest.raises(KeyError):
            dkt.get_position("k")

        TestDict.const_item_dict["l"] = 1
        assert dkt.get_position("l") == 5


class TestLoadBackup:
    @staticmethod
    def block_exec(self, *args, **kwargs):
        raise RuntimeError("aa")

    def test_no_backup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(load_backup.state_store, "save_folder", tmp_path)
        monkeypatch.setattr(load_backup.QMessageBox, "exec_", self.block_exec)
        monkeypatch.setattr(load_backup.QMessageBox, "question", self.block_exec)
        monkeypatch.setattr(load_backup, "parsed_version", parse("0.13.13"))
        load_backup.import_config()

        monkeypatch.setattr(load_backup.state_store, "save_folder", tmp_path / "0.13.13")
        load_backup.import_config()

    def test_no_backup_old(self, monkeypatch, tmp_path):
        monkeypatch.setattr(load_backup.state_store, "save_folder", tmp_path / "0.13.13")
        monkeypatch.setattr(load_backup.QMessageBox, "exec_", self.block_exec)
        monkeypatch.setattr(load_backup.QMessageBox, "question", self.block_exec)
        monkeypatch.setattr(load_backup, "parsed_version", parse("0.13.13"))
        (tmp_path / "0.13.14").mkdir()
        (tmp_path / "0.13.15").mkdir()

        load_backup.import_config()

    @pytest.mark.parametrize("response", [load_backup.QMessageBox.Yes, load_backup.QMessageBox.No])
    def test_older_exist(self, monkeypatch, tmp_path, response):
        def create_file(file_path: Path):
            file_path.parent.mkdir()
            with open(file_path, "w") as f:
                print(1, file=f)

        def question(*args, **kwargs):
            return response

        monkeypatch.setattr(load_backup.state_store, "save_folder", tmp_path / "0.13.13")
        monkeypatch.setattr(load_backup.QMessageBox, "exec_", self.block_exec)
        monkeypatch.setattr(load_backup.QMessageBox, "question", question)
        monkeypatch.setattr(load_backup, "napari_get_settings", lambda x: NapariSettingsMock)
        create_file(tmp_path / "0.13.14" / "14.txt")
        create_file(tmp_path / "0.13.12" / "12.txt")
        create_file(tmp_path / "0.13.10" / "10.txt")
        load_backup.import_config()
        if response == QMessageBox.Yes:
            assert (tmp_path / "0.13.13" / "12.txt").exists()
        else:
            assert not (tmp_path / "0.13.13").exists()


class NapariSettingsMock:
    @staticmethod
    def load():
        return 1


@pytest.fixture
def image(tmp_path):
    data = np.random.random((10, 10, 2))
    return Image(data=data, image_spacing=(10, 10), axes_order="XYC", file_path=str(tmp_path / "test.tiff"))


@pytest.fixture
def roi():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:-2, 2:5] = 1
    data[2:-2, 5:-2] = 2
    return data


class TestBaseSettings:
    def test_image_settings(self, tmp_path, image, roi, qtbot):
        widget = base_settings.QWidget()
        qtbot.addWidget(widget)
        settings = base_settings.ImageSettings()
        settings.set_parent(widget)
        assert settings.image_spacing == ()
        assert settings.image_shape == ()
        assert settings.image_path == ""
        assert settings.channels == 0
        settings.image = image
        assert settings.image_spacing == image.spacing
        assert settings.image_path == str(tmp_path / "test.tiff")
        assert settings.image_shape == (1, 1, 10, 10, 2)
        with qtbot.waitSignal(settings.image_spacing_changed):
            settings.image_spacing = (7, 7)
        assert image.spacing == (7, 7)
        with qtbot.waitSignal(settings.image_spacing_changed):
            settings.image_spacing = (1, 8, 8)
        assert image.spacing == (8, 8)
        with pytest.raises(ValueError):
            settings.image_spacing = (6,)

        assert settings.is_image_2d()
        assert settings.has_channels
        with qtbot.waitSignal(settings.image_changed[str]):
            settings.image_path = str(tmp_path / "test2.tiff")
        assert image.file_path == str(tmp_path / "test2.tiff")

        with pytest.raises(ValueError, match="roi do not fit to image"):
            settings.roi = roi[:-2]
        with qtbot.waitSignal(settings.roi_changed):
            settings.roi = roi
        assert all(settings.sizes == [64, 18, 18])
        assert all(settings.components_mask() == [0, 1, 1])

        assert tuple(settings.roi_info.bound_info.keys()) == (1, 2)

        with qtbot.waitSignal(settings.roi_clean):
            settings.roi = None

        settings.roi = ROIInfo(roi)

        settings.image = None
        assert settings.image is not None

    def test_view_settings(self, tmp_path, image, roi, qtbot):
        settings = base_settings.ViewSettings()
        assert settings.theme_name == "light"
        assert hasattr(settings.theme, "text")
        assert isinstance(settings.style_sheet, str)
        with pytest.raises(ValueError):
            settings.theme_name = "aaaa"
        with qtbot.assertNotEmitted(settings.theme_changed):
            settings.theme_name = "light"
        with qtbot.waitSignal(settings.theme_changed):
            settings.theme_name = "dark"
        assert settings.theme_name == "dark"
        assert isinstance(settings.theme_list(), (tuple, list))

        assert settings.chosen_colormap == base_settings.starting_colors
        with qtbot.waitSignal(settings.colormap_changes):
            settings.chosen_colormap = ["A", "B"]
        assert settings.chosen_colormap == base_settings.starting_colors
        assert settings.current_labels == "default"
        with pytest.raises(ValueError):
            settings.current_labels = "a"

    def test_colormap_dict(self):
        colormap_dict = base_settings.ColormapDict({})
        assert colormap_dict.colormap_removed == colormap_dict.item_removed
        assert colormap_dict.colormap_added == colormap_dict.item_added

    def test_label_color_dict(self):
        label_dict = base_settings.LabelColorDict({})
        assert isinstance(label_dict.get_array("default"), np.ndarray)

    def test_base_settings(self, tmp_path):
        settings = base_settings.BaseSettings(tmp_path / "data.json")
        assert settings.theme_name == "light"
