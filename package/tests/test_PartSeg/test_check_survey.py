from io import BytesIO
from unittest.mock import Mock

import pytest

from PartSeg._launcher.check_survey import CheckSurveyThread, SurveyMessageBox


@pytest.fixture(autouse=True)
def _mock_file_path(monkeypatch, tmp_path):
    monkeypatch.setattr("PartSeg._launcher.check_survey.IGNORE_FILE_PATH", tmp_path / "file1.txt")


def test_check_survey_thread(qtbot, monkeypatch, tmp_path):
    urlopen_mock = Mock(return_value=BytesIO(b"some data"))
    monkeypatch.setattr("urllib.request.urlopen", urlopen_mock)
    thr = CheckSurveyThread("test_url")

    thr.run()
    assert urlopen_mock.call_count == 1
    assert thr.survey_url == "some data"


def test_survey_message_box(qtbot):
    msg = SurveyMessageBox("test_url")
    qtbot.addWidget(msg)
