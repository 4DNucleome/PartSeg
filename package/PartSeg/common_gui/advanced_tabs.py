"""
This module contains base for advanced window for PartSeg.
In this moment controlling colormaps tabs and developer PartSegCore
"""
import importlib
import sys
from functools import partial
from typing import List

from qtpy.QtCore import QByteArray, Qt
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QColorDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PartSeg import plugins
from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.common_gui.colormap_creator import PColormapCreator, PColormapList
from PartSeg.common_gui.label_create import ColorShow, LabelChoose, LabelEditor
from PartSegCore import register, state_store


class DevelopTab(QWidget):
    """
    Widget for developer utilities. Currently only contains button for reload algorithms and

    To enable it run program with `--develop` flag.

    If you would like to use it for developing your own algorithm and modify same of ParsSeg class
    please protect this part of code with something like:

    >>> if tifffile.tifffile.TiffPage.__module__ != "PartSegImage.image_reader":

    This is taken from :py:mod:`PartSegImage.image_reader`
    """

    def __init__(self):
        super().__init__()

        # noinspection PyArgumentList
        self.reload_btm = QPushButton("Reload algorithms", clicked=self.reload_algorithm_action)
        layout = QGridLayout()
        layout.addWidget(self.reload_btm, 0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        self.setLayout(layout)

    def reload_algorithm_action(self):
        """Function for reload plugins and algorithms"""
        for val in register.reload_module_list:
            print(val, file=sys.stderr)
            importlib.reload(val)
        for el in plugins.get_plugins():
            importlib.reload(el)
        importlib.reload(register)
        importlib.reload(plugins)
        plugins.register()
        for el in self.parent().parent().reload_list:
            el()


class MaskControl(QWidget):
    def __init__(self, settings: ViewSettings):
        super().__init__()
        self.settings = settings
        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.Widget)
        self.color_picker.setOptions(QColorDialog.DontUseNativeDialog | QColorDialog.NoButtons)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0, 1)
        self.opacity_spin.setSingleStep(0.1)
        self.opacity_spin.setDecimals(2)
        self.change_mask_color_btn = QPushButton("Change mask color")
        self.current_mask_color_preview = ColorShow(
            self.settings.get_from_profile("mask_presentation_color", [255, 255, 255])
        )

        self.opacity_spin.setValue(self.settings.get_from_profile("mask_presentation_opacity", 1))

        self.current_mask_color_preview.setAutoFillBackground(True)
        self.change_mask_color_btn.clicked.connect(self.change_color)
        self.opacity_spin.valueChanged.connect(self.change_opacity)

        layout = QVBoxLayout()
        layout.addWidget(self.color_picker)
        layout2 = QHBoxLayout()
        layout2.addWidget(self.change_mask_color_btn)
        layout2.addWidget(self.current_mask_color_preview, 1)
        layout2.addWidget(QLabel("Mask opacity"))
        layout2.addWidget(self.opacity_spin)
        layout.addLayout(layout2)
        self.setLayout(layout)

    def set_color_preview(self, color):
        self.current_mask_color_preview.set_color(color)

    def change_color(self):
        color = self.color_picker.currentColor()
        color = (color.red(), color.green(), color.blue())
        self.settings.set_in_profile("mask_presentation_color", color)
        self.settings.mask_representation_changed_emit()
        self.set_color_preview(color)

    def change_opacity(self):
        self.settings.set_in_profile("mask_presentation_opacity", self.opacity_spin.value())
        self.settings.mask_representation_changed_emit()


class ColorControl(QTabWidget):
    """
    Class for storage all settings for labels and colormaps.
    """

    def __init__(self, settings: ViewSettings, image_view_names: List[str]):
        super().__init__()
        self.colormap_selector = PColormapCreator(settings)
        self.color_preview = PColormapList(settings, image_view_names)
        self.color_preview.edit_signal.connect(self.colormap_selector.set_colormap)
        self.color_preview.edit_signal.connect(partial(self.setCurrentWidget, self.colormap_selector))
        self.label_editor = LabelEditor(settings)
        self.label_view = LabelChoose(settings)
        self.label_view.edit_signal.connect(partial(self.setCurrentWidget, self.label_editor))
        self.label_view.edit_signal[list].connect(self.label_editor.set_colors)
        self.mask_control = MaskControl(settings)
        self.addTab(self.color_preview, "Color maps")
        self.addTab(self.colormap_selector, "Color Map creator")
        self.addTab(self.label_view, "Select labels")
        self.addTab(self.label_editor, "Create labels")
        self.addTab(self.mask_control, "Mask marking")


class AdvancedWindow(QTabWidget):
    """
    Base class for advanced windows.
    It contains colormap management connected tabs: :py:class:`.PColormapCreator`
    and :py:class:`.PColormapList`.

    :param settings: program settings
    :param image_view_names: passed as second argument to :py:class:`~.PColormapList`
    """

    def __init__(self, settings: ViewSettings, image_view_names: List[str], parent=None):
        super().__init__(parent)
        self.color_control = ColorControl(settings, image_view_names)
        self.settings = settings

        self.develop = DevelopTab()
        self.addTab(self.color_control, "Color control")
        if state_store.develop:
            self.addTab(self.develop, "Develop")
        if self.window() == self:
            try:
                geometry = self.settings.get_from_profile("advanced_window_geometry")
                self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))
            except KeyError:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Save geometry if widget is used as standalone window.
        """
        if self.window() == self:
            self.settings.set_in_profile("advanced_window_geometry", self.saveGeometry().toHex().data().decode("ascii"))
        super().closeEvent(event)
