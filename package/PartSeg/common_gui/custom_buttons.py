import qtawesome as qta
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QPushButton

from PartSeg.common_backend.base_settings import BaseSettings


class QtViewerPushButton(QPushButton):
    def __init__(self, icon: str, settings: BaseSettings):
        super().__init__(qta.icon(icon), "")
        self.settings = settings
        settings.theme_changed.connect(self._theme_changed)
        self._theme_changed()

    def _theme_changed(self):
        color = self.settings.theme.text
        self.setIcon(qta.icon("fa5s.search", color=QColor(*color.as_rgb_tuple())))


class SearchROIButton(QtViewerPushButton):
    def __init__(self, settings: BaseSettings):
        super().__init__("fa5s.search", settings)
        self.setToolTip("Search component")
