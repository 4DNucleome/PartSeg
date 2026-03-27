import json
import urllib.error
import urllib.request
from datetime import date, timedelta
from io import StringIO
from unittest.mock import MagicMock

import packaging.version
import pytest
from qtpy.QtWidgets import QMessageBox

from PartSeg import state_store
from PartSeg._launcher import check_version
from PartSeg._launcher.check_version import IGNORE_FILE


@pytest.mark.enablethread
@pytest.mark.parametrize("thread", [False, True])
@pytest.mark.parametrize("package_name", ["PartSeg", "sample_name"])
def test_fetching(thread, package_name, monkeypatch, qtbot):
    def urlopen_mock(url):
        assert f"https://pypi.org/pypi/{package_name}/json" == url
        return StringIO(
            json.dumps(
                {"info": {"version": "0.10.0", "home_page": f"https://4dnucleome.cent.uw.edu.pl/{package_name}/"}}
            )
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    assert packaging.version.parse("0.10.0") < packaging.version.parse("0.11.0")
    chk_thr = check_version.CheckVersionThread(package_name, base_version="0.11.0")
    chk_thr.release = "0.10.0"
    if thread:
        with qtbot.wait_signal(chk_thr.finished):
            chk_thr.start()
    else:
        chk_thr.run()
    assert chk_thr.release == "0.10.0"
    assert chk_thr.url == f"https://4dnucleome.cent.uw.edu.pl/{package_name}/"
    chk_thr.deleteLater()


@pytest.mark.parametrize("frozen", [True, False])
def test_show_window_dialog(monkeypatch, frozen, qtbot):
    values = ["", ""]

    class MockMessageBox:
        StandardButton = QMessageBox.StandardButton

        Icon = QMessageBox.Icon

        def __init__(self, _type, title, message, _buttons):
            values[0] = title
            values[1] = message

        @staticmethod
        def exec_():
            return check_version.QMessageBox.StandardButton.Ok

    chk_thr = check_version.CheckVersionThread(base_version="0.10.0")
    chk_thr.release = "0.11.0"
    monkeypatch.setattr(check_version.sys, "frozen", frozen, raising=False)
    monkeypatch.setattr(check_version, "QMessageBox", MockMessageBox)
    chk_thr.show_version_info()
    assert values[0] == "New release"
    if frozen:
        assert "You can download next release form" in values[1]
    else:
        assert "You can update it from pypi" in values[1]


def test_no_update(monkeypatch, qtbot, tmp_path):
    # check if nothing is reported on ulr error
    monkeypatch.setattr("PartSeg.state_store.save_folder", tmp_path)

    def urlopen_mock(url):
        raise urllib.error.URLError("test purpose")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    chk_thr = check_version.CheckVersionThread(base_version="0.10.0")
    chk_thr.run()
    assert chk_thr.release == "0.10.0"


def test_error_report(monkeypatch, qtbot):
    sentry_val = [False]

    def urlopen_mock(url):
        raise RuntimeError

    def sentry_mock(e):
        sentry_val[0] = True

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)
    monkeypatch.setattr(check_version.sentry_sdk, "capture_exception", sentry_mock)
    chk_thr = check_version.CheckVersionThread()
    chk_thr.run()

    assert sentry_val[0] is True


def test_ignore_file_exists(monkeypatch, qtbot, tmp_path):
    with (tmp_path / IGNORE_FILE).open("w") as f_p:
        f_p.write(date.today().isoformat())

    monkeypatch.setattr(state_store, "save_folder", tmp_path)

    def urlopen_mock(url):
        raise RuntimeError

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)

    chk_thr = check_version.CheckVersionThread()
    chk_thr.run()


def test_ignore_file_exists_old_date(monkeypatch, qtbot, tmp_path):
    with (tmp_path / IGNORE_FILE).open("w") as f_p:
        f_p.write((date.today() - timedelta(days=60)).isoformat())

    monkeypatch.setattr(state_store, "save_folder", tmp_path)

    def urlopen_mock(_url):
        return StringIO('{"info": {"version": "0.11.0", "home_page": "example.org"}}')

    monkeypatch.setattr(urllib.request, "urlopen", urlopen_mock)

    chk_thr = check_version.CheckVersionThread(base_version="0.10.0")
    chk_thr.run()
    assert not (tmp_path / IGNORE_FILE).exists()
    assert chk_thr.release == "0.11.0"
    assert chk_thr.url == "example.org"


def test_create_ignore(qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(state_store, "save_folder", tmp_path)

    chk_thr = check_version.CheckVersionThread(base_version="0.10.0")
    chk_thr.release = "0.11.0"
    monkeypatch.setattr(check_version.QMessageBox, "exec_", MagicMock(return_value=check_version.QMessageBox.Ignore))
    chk_thr.show_version_info()
    assert (tmp_path / IGNORE_FILE).exists()
    with (tmp_path / IGNORE_FILE).open() as f_p:
        assert f_p.read() == date.today().isoformat()
