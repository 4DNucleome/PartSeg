import sys

import pytest
import sentry_sdk
from qtpy.QtWidgets import QMessageBox

from PartSeg.common_backend import except_hook
from PartSeg.common_gui.error_report import ErrorDialog
from PartSegCore import state_store
from PartSegCore.segmentation.algorithm_base import SegmentationLimitException
from PartSegImage import TiffFileException


def test_show_error(monkeypatch):
    exec_list = []

    def exec_mock(self):
        exec_list.append(self)

    monkeypatch.setattr(except_hook.QMessageBox, "exec", exec_mock)
    monkeypatch.setattr(ErrorDialog, "exec", exec_mock)

    except_hook.show_error()
    assert exec_list == []
    except_hook.show_error(TiffFileException("sample"))
    assert len(exec_list) == 1, "exec not called"
    message = exec_list[0]
    assert isinstance(message, QMessageBox)
    assert message.icon() == QMessageBox.Critical
    assert message.windowTitle() == "Tiff error"
    assert message.text().startswith("During read file there is an error")

    exec_list = []
    except_hook.show_error(SegmentationLimitException("sample"))
    assert len(exec_list) == 1, "exec not called"
    message = exec_list[0]
    assert isinstance(message, QMessageBox)
    assert message.icon() == QMessageBox.Critical
    assert message.windowTitle() == "Segmentation limitations"
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
def test_show_warning(monkeypatch, header, text):
    exec_list = []

    def exec_mock(self):
        exec_list.append(self)

    monkeypatch.setattr(except_hook.QMessageBox, "exec", exec_mock)
    except_hook.show_warning(header, text)
    assert len(exec_list) == 1


def test_my_excepthook(monkeypatch, capsys):
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

    monkeypatch.setattr(sys, "__excepthook__", excepthook_catch)
    monkeypatch.setattr(sys, "exit", exit_catch)
    monkeypatch.setattr(sentry_sdk, "capture_exception", capture_exception_catch)
    monkeypatch.setattr(except_hook, "show_error", show_error_catch)
    monkeypatch.setattr(except_hook, "parsed_version", ParsedVersionMockFalse)

    monkeypatch.setattr(state_store, "show_error_dialog", False)
    except_hook.my_excepthook(KeyboardInterrupt, KeyboardInterrupt(), [])
    assert exit_list == [1]
    assert capsys.readouterr().err == "KeyboardInterrupt close\n"

    except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
    assert len(catch_list) == 1
    assert catch_list[0][0] is RuntimeError

    monkeypatch.setattr(state_store, "show_error_dialog", True)

    except_hook.my_excepthook(KeyboardInterrupt, KeyboardInterrupt(), [])
    assert exit_list == [1, 1]
    assert capsys.readouterr().err == "KeyboardInterrupt close\n"

    except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
    assert len(error_list) == 1
    assert isinstance(error_list[0], RuntimeError)
    assert len(sentry_catch_list) == 0

    monkeypatch.setattr(state_store, "report_errors", True)
    monkeypatch.setattr(except_hook, "parsed_version", ParsedVersionMockTrue)
    except_hook.my_excepthook(RuntimeError, RuntimeError("aaa"), [])
    assert len(sentry_catch_list) == 1
    assert isinstance(sentry_catch_list[0], RuntimeError)


class ParsedVersionMockTrue:
    is_devrelease = True


class ParsedVersionMockFalse:
    is_devrelease = False
