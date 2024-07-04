import urllib.error
import urllib.request
from contextlib import suppress

from qtpy.QtCore import QThread
from qtpy.QtWidgets import QMessageBox
from superqt import ensure_main_thread

IGNORE_DAYS = 21
IGNORE_FILE = "ignore_survey.txt"


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
        """This function perform check"""
        with suppress(urllib.error.URLError), urllib.request.urlopen(self.survey_file_url) as r:  # nosec  # noqa: S310
            self.survey_url = r.read().decode("utf-8").strip()

    @ensure_main_thread
    def show_version_info(self):
        if self.survey_url:
            QMessageBox.information(
                None,
                "PartSeg survey",
                f"Please fill the survey to help us improve PartSeg: {self.survey_url}",
                QMessageBox.Ok,
            )
