import json
import urllib.request
from io import StringIO

import pytest

from PartSeg._launcher import check_version


@pytest.mark.enablethread
@pytest.mark.parametrize("package_name", ["PartSeg", "sample_name"])
@pytest.mark.parametrize("thread", [True, False])
def test_fetching(package_name, monkeypatch, qtbot, thread):
    def urlopen_mock(url):
        assert f"https://pypi.org/pypi/{package_name}/json" == url
        return StringIO(
            json.dumps(
                {"info": {"version": "0.10.0", "home_page": f"https://4dnucleome.cent.uw.edu.pl/{package_name}/"}}
            )
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    chk_thr = check_version.CheckVersionThread(package_name, base_version="0.11.0")
    if thread:
        with qtbot.wait_signal(chk_thr.finished):
            chk_thr.start()
    else:
        chk_thr.run()
    assert chk_thr.release == "0.10.0"
    assert chk_thr.url == f"https://4dnucleome.cent.uw.edu.pl/{package_name}/"


@pytest.mark.parametrize("frozen", [True, False])
def test_show_window_dialog(monkeypatch, frozen, qtbot):
    values = ["", ""]

    class MockMessageBox:
        Information = 1
        Ok = 2

        def __init__(self, _type, title, message, _buttons):
            values[0] = title
            values[1] = message

        @staticmethod
        def exec():
            return True

    chk_thr = check_version.CheckVersionThread(base_version="0.11.0")
    monkeypatch.setattr(check_version, "__version__", "0.10.0")
    monkeypatch.setattr(check_version.sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(check_version, "QMessageBox", MockMessageBox)
    chk_thr.show_version_info()
    assert values[0] == "New release"
    if frozen:
        assert "You can download next release form" in values[1]
    else:
        assert "You can update it from pypi" in values[1]


def test_no_update(monkeypatch, qtbot):
    monkeypatch.setattr(check_version.state_store, "check_for_updates", False)

    def urlopen_mock(url):
        raise RuntimeError()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    chk_thr = check_version.CheckVersionThread()
    chk_thr.run()


def test_error_report(monkeypatch, qtbot):
    sentry_val = [False]

    def urlopen_mock(url):
        raise RuntimeError()

    def sentry_mock(e):
        sentry_val[0] = True

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    monkeypatch.setattr(check_version.sentry_sdk, "capture_exception", sentry_mock)
    chk_thr = check_version.CheckVersionThread()
    chk_thr.run()

    assert sentry_val[0] is True
