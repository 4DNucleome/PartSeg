import argparse
import sys
import typing
from typing import Callable, Optional

import numpy as np
import pytest
import sentry_sdk
from packaging.version import parse
from qtpy.QtWidgets import QMessageBox

from PartSeg.common_backend import base_argparser, except_hook, progress_thread, segmentation_thread
from PartSeg.common_gui.error_report import ErrorDialog
from PartSegCore import state_store
from PartSegCore.algorithm_describe_base import AlgorithmProperty, ROIExtractionProfile
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
