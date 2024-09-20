from importlib.metadata import version
from typing import List, Sequence

from napari import Viewer
from napari.layers import Labels
from packaging.version import parse as parse_version
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QTabWidget

from PartSeg.common_backend.base_settings import BaseSettings
from PartSeg.common_gui.label_create import LabelChoose, LabelEditor, LabelShow
from PartSeg.plugins.napari_widgets._settings import get_settings

NAPARI_GE_5_0 = parse_version(version("napari")) >= parse_version("0.5.0a1")
NAPARI_GE_4_19 = parse_version(version("napari")) >= parse_version("0.4.19a1")


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
        if len(self.viewer.layers.selection) == 1 and isinstance(next(iter(self.viewer.layers.selection)), Labels):
            self.apply_label_btn.setEnabled(True)
            self.apply_label_btn.setToolTip("Apply labels to selected layer")
        else:
            self.apply_label_btn.setEnabled(False)
            self.apply_label_btn.setToolTip("Select one labels layer to apply custom labels")

    def apply_label(self):
        if len(self.viewer.layers.selection) == 1 and isinstance(
            layer := next(iter(self.viewer.layers.selection)), Labels
        ):
            max_val = layer.data.max()
            labels = {i + 1: [x / 255 for x in self.label[i % len(self.label)]] for i in range(max_val + 5)}
            labels[None] = [0, 0, 0, 0]
            if NAPARI_GE_4_19:
                from napari.utils.colormaps import DirectLabelColormap

                layer.colormap = DirectLabelColormap(color_dict=labels)
            else:
                layer.color = labels


class NaparliLabelChoose(LabelChoose):
    def __init__(self, viewer: Viewer, settings, parent=None):
        super().__init__(settings, parent)
        self.viewer = viewer

    def _label_show(self, name: str, label: List[Sequence[float]], removable) -> LabelShow:
        return NapariLabelShow(self.viewer, name, label, removable, self)


class NapariLabelEditor(LabelEditor):
    settings: BaseSettings

    def save(self):
        super().save()
        self.settings.dump()


class LabelSelector(QTabWidget):
    def __init__(self, viewer: Viewer, parent=None):
        super().__init__(parent)
        self.viewer = viewer
        settings = get_settings()
        self.settings = settings
        self.label_editor = NapariLabelEditor(settings)
        self.label_view = NaparliLabelChoose(viewer, settings)
        self.addTab(self.label_view, "Select labels")
        self.addTab(self.label_editor, "Create labels")

        self.label_view.edit_with_name_signal.connect(self.label_editor.set_colors)
        self.label_view.edit_signal.connect(self._set_label_editor)

    def _set_label_editor(self):
        self.setCurrentWidget(self.label_editor)
