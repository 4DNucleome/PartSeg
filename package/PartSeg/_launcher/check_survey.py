import os
import time
import urllib.error
import urllib.request
import webbrowser
from contextlib import suppress
from pathlib import Path

from qtpy.QtCore import Qt, QThread
from qtpy.QtWidgets import QMessageBox, QWidget
from superqt import ensure_main_thread

from PartSeg import state_store

IGNORE_DAYS = 21
IGNORE_FILE = "ignore_survey.txt"

IGNORE_FILE_PATH = (Path(state_store.save_folder) / ".." / IGNORE_FILE).resolve()


class SurveyMessageBox(QMessageBox):
    def __init__(self, survey_url: str, parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("PartSeg survey")
        self.setIcon(QMessageBox.Icon.Information)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setText(
            f'Please fill the survey <a href="{survey_url}">{survey_url}</a> to <b>help us</b> improve PartSeg'
        )
        self._open_btn = self.addButton("Open survey", QMessageBox.ButtonRole.AcceptRole)
        self._close_btn = self.addButton("Close", QMessageBox.ButtonRole.RejectRole)
        self._ignore_btn = self.addButton("Ignore", QMessageBox.ButtonRole.DestructiveRole)
        self.setEscapeButton(self._ignore_btn)
        self.survey_url = survey_url

        self._open_btn.clicked.connect(self._open_survey)
        self._ignore_btn.clicked.connect(self._ignore_survey)

    def _open_survey(self):
        webbrowser.open(self.survey_url)

    def _ignore_survey(self):
        with IGNORE_FILE_PATH.open("w") as f_p:
            f_p.write(self.survey_url)

    def exec_(self):
        result = super().exec_()
        if not IGNORE_FILE_PATH.parent.exists():
            IGNORE_FILE_PATH.parent.mkdir(parents=True)
        IGNORE_FILE_PATH.touch()
        return result


class CheckSurveyThread(QThread):
    """Thread to check if there is new PartSeg release. Checks base on newest version available on pypi_

    .. _PYPI: https://pypi.org/project/PartSeg/
    """

    def __init__(self, survey_file_url="https://raw.githubusercontent.com/4DNucleome/PartSeg/develop/survey_url.txt"):
        super().__init__()
        self.survey_file_url = survey_file_url
        self.survey_url = ""
        self.finished.connect(self.show_version_info)

    def run(self):
        """This function perform check if there is any active survey."""
        if IGNORE_FILE_PATH.exists() and (time.time() - os.path.getmtime(IGNORE_FILE_PATH)) < 60 * 60 * 16:
            return
        with suppress(urllib.error.URLError), urllib.request.urlopen(self.survey_file_url) as r:  # nosec  # noqa: S310
            self.survey_url = r.read().decode("utf-8").strip()

    @ensure_main_thread
    def show_version_info(self):
        if os.path.exists(IGNORE_FILE_PATH):
            with open(IGNORE_FILE_PATH) as f_p:
                old_survey_link = f_p.read().strip()
            if old_survey_link == self.survey_url:
                return

        if self.survey_url:
            SurveyMessageBox(self.survey_url).exec_()
