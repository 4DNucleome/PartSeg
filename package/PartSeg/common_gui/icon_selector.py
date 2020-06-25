import os

from qtpy.QtGui import QIcon

from PartSegData import icons_dir


class IconSelector:
    def __init__(self):
        self._close_icon = None
        self._edit_icon = None

    @property
    def close_icon(self) -> QIcon:
        if self._close_icon is None:
            self._close_icon = QIcon(os.path.join(icons_dir, "task-reject.png"))
        return self._close_icon

    @property
    def edit_icon(self):
        if self._edit_icon is None:
            self._edit_icon = QIcon(os.path.join(icons_dir, "configure.png"))
        return self._edit_icon
