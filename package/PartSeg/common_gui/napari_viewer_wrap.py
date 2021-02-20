from typing import Optional

from napari import Viewer as NViewer
from qtpy.QtWidgets import QCheckBox

from PartSeg.common_backend.base_settings import BaseSettings


class Viewer(NViewer):
    _napari_app_id = False

    def __init__(self, settings: Optional[BaseSettings] = None, **kwargs):
        super().__init__(**kwargs)
        self.settings = settings
        self.sync_chk = QCheckBox("Enable layer sync")
        self.window.add_dock_widget(self.sync_chk, area="left")
