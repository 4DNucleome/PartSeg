from io import BytesIO
from unittest.mock import MagicMock, Mock

import pytest

from PartSeg._launcher.check_survey import CheckSurveyThread, SurveyMessageBox


@pytest.fixture(autouse=True)
def _mock_file_path(monkeypatch, tmp_path):
    monkeypatch.setattr("PartSeg._launcher.check_survey.IGNORE_FILE_PATH", tmp_path / "file1.txt")


@pytest.mark.usefixtures("qtbot")
def test_check_survey_thread(monkeypatch, tmp_path):
    urlopen_mock = Mock(return_value=BytesIO(b"some data"))
    monkeypatch.setattr("urllib.request.urlopen", urlopen_mock)
    thr = CheckSurveyThread("test_url")

    thr.run()
    assert urlopen_mock.call_count == 1
    assert thr.survey_url == "some data"


@pytest.mark.usefixtures("qtbot")
def test_thread_ignore_file_exist(monkeypatch, tmp_path):
    message_mock = Mock()
    monkeypatch.setattr("PartSeg._launcher.check_survey.SurveyMessageBox", message_mock)
    with (tmp_path / "file1.txt").open("w") as f_p:
        f_p.write("test_url")

    thr = CheckSurveyThread("test_url2")
    thr.survey_url = "test_url"

    thr.show_version_info()
    assert message_mock.call_count == 0


@pytest.mark.usefixtures("qtbot")
def test_call_survey_message_box(monkeypatch):
    message_mock = MagicMock()
    monkeypatch.setattr("PartSeg._launcher.check_survey.SurveyMessageBox", message_mock)
    thr = CheckSurveyThread("test_url")
    thr.survey_url = "test_url"
    thr.show_version_info()
    assert message_mock.call_count == 1


def test_survey_message_box(qtbot, monkeypatch):
    msg = SurveyMessageBox("test_url")
    qtbot.addWidget(msg)
