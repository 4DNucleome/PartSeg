from typing import List, Sequence

from napari import Viewer
from napari.layers import Labels
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QTabWidget

from PartSeg.common_gui.label_create import LabelChoose, LabelEditor, LabelShow
from PartSeg.plugins.napari_widgets._settings import get_settings


class NapariLabelShow(LabelShow):
    def __init__(self, viewer: Viewer, name: str, label: List[Sequence[float]], removable, parent=None):
        super().__init__(name, label, removable, parent)
        self.viewer = viewer

        self.apply_label_btn = QPushButton("Apply")

        layout: QHBoxLayout = self.layout()
        layout.removeWidget(self.radio_btn)
        layout.insertWidget(0, self.apply_label_btn)
        viewer.layers.selection.events.changed.connect(self.update_preview)
        self.apply_label_btn.clicked.connect(self.apply_label)

    def update_preview(self, _event=None):
        if len(self.viewer.layers.selection) == 1 and isinstance(list(self.viewer.layers.selection)[0], Labels):
            self.apply_label_btn.setEnabled(True)
            self.apply_label_btn.setToolTip("Apply labels to selected layer")
        else:
            self.apply_label_btn.setEnabled(False)
            self.apply_label_btn.setToolTip("Select one labels layer to apply custom labels")

    def apply_label(self):
        if len(self.viewer.layers.selection) == 1 and isinstance(
            layer := list(self.viewer.layers.selection)[0], Labels
        ):
            max_val = layer.data.max()
            labels = {i + 1: [x / 255 for x in self.label[i % len(self.label)]] for i in range(max_val + 5)}
            layer.color = labels


class NaparliLabelChoose(LabelChoose):
    def __init__(self, viewer: Viewer, settings, parent=None):
        super().__init__(settings, parent)
        self.viewer = viewer

    def _label_show(self, name: str, label: List[Sequence[float]], removable) -> LabelShow:
        return NapariLabelShow(self.viewer, name, label, removable, self)


class LabelSelector(QTabWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        settings = get_settings()
        self.settings = settings
        self.label_editor = LabelEditor(settings)
        self.label_view = NaparliLabelChoose(viewer, settings)
        self.addTab(self.label_view, "Select labels")
        self.addTab(self.label_editor, "Create labels")
