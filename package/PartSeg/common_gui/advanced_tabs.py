"""
This module contains base for the advanced window for PartSeg.
At this moment controlling colormaps tabs and developer PartSegCore
"""

import importlib
import logging
from contextlib import suppress
from typing import List

from qtpy.QtCore import QByteArray, Qt
from qtpy.QtGui import QCloseEvent
from qtpy.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from PartSeg import plugins, state_store
from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg.common_backend.base_settings import BaseSettings, ViewSettings
from PartSeg.common_gui.colormap_creator import PColormapCreator, PColormapList
from PartSeg.common_gui.dict_viewer import DictViewer
from PartSeg.common_gui.label_create import ColorShow, LabelChoose, LabelEditor
from PartSeg.common_gui.universal_gui_part import CustomDoubleSpinBox
from PartSegCore import plugins as core_plugins
from PartSegCore import register

RENDERING_LIST = ["iso_categorical", "translucent"]

RENDERING_MODE_NAME_STR = "rendering_mode"
SEARCH_ZOOM_FACTOR_STR = "search_zoom_factor"


class DevelopTab(QWidget):
    """
    Widget for developer utilities. Currently only contains button for reload algorithms and

    To enable it run program with a `--develop` flag.

    If you would like to use it for developing your own algorithm and modify some of the PartSeg class.
    Please protect this part of code with something like:

    >>> if tifffile.tifffile.TiffPage.__module__ != "PartSegImage.image_reader":

    This is taken from :py:mod:`PartSegImage.image_reader`
    """

    def __init__(self, parent=None):
        super().__init__(parent=parent)

        # noinspection PyArgumentList
        self.reload_btn = QPushButton("Reload algorithms", clicked=self.reload_algorithm_action)
        layout = QGridLayout(self)
        layout.addWidget(self.reload_btn, 0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(1, 1)
        self.setLayout(layout)

    def reload_algorithm_action(self):
        """Function for reload plugins and algorithms"""
        msg = "Reloading %(mod_name)"
        for val in register.reload_module_list:
            logging.info(msg, extra={"mod_name": val.__name__})
            importlib.reload(val)
        for el in plugins.get_plugins():
            logging.info(msg, extra={"mod_name": el.__name__})
            importlib.reload(el)
        for el in core_plugins.get_plugins():
            logging.info(msg, extra={"mod_name": el.__name__})
            importlib.reload(el)
        importlib.reload(register)
        importlib.reload(plugins)
        importlib.reload(core_plugins)
        plugins.register()
        core_plugins.register()
        for el in self.parent().parent().reload_list:
            el()


class MaskControl(QWidget):
    def __init__(self, settings: ViewSettings, parent=None):
        super().__init__(parent=parent)
        self.settings = settings
        self.color_picker = QColorDialog()
        self.color_picker.setWindowFlag(Qt.WindowType.Widget)
        self.color_picker.setOptions(
            QColorDialog.ColorDialogOption.DontUseNativeDialog | QColorDialog.ColorDialogOption.NoButtons
        )
        self.opacity_spin = QDoubleSpinBox(self)
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
        self.set_color_preview(color)

    def change_opacity(self):
        self.settings.set_in_profile("mask_presentation_opacity", self.opacity_spin.value())


class Appearance(QWidget):
    def __init__(self, settings: ViewSettings):
        super().__init__()
        self.settings = settings

        self.layout_list = QComboBox()
        self.layout_list.addItems(self.settings.theme_list())
        self.layout_list.setCurrentText(self.settings.theme_name)
        self.layout_list.currentIndexChanged.connect(self.change_theme)

        self.labels_render_cmb = QComboBox()
        self.labels_render_cmb.addItems(RENDERING_LIST)
        self._update_render_mode()
        self.labels_render_cmb.currentTextChanged.connect(self.change_render_mode)
        settings.connect_to_profile(RENDERING_MODE_NAME_STR, self._update_render_mode)

        self.zoom_factor_spin_box = CustomDoubleSpinBox()
        self.zoom_factor_spin_box.setValue(settings.get_from_profile(SEARCH_ZOOM_FACTOR_STR, 1.2))
        self.zoom_factor_spin_box.valueChanged.connect(self.change_zoom_factor)
        settings.connect_to_profile(SEARCH_ZOOM_FACTOR_STR, self._update_zoom_factor)

        self.scale_bar_ticks = QCheckBox()
        self.scale_bar_ticks.setChecked(settings.get_from_profile("scale_bar_ticks", True))
        self.scale_bar_ticks.stateChanged.connect(self.change_scale_bar_ticks)
        settings.connect_to_profile("scale_bar_ticks", self._update_scale_bar_ticks)

        layout = QGridLayout()
        layout.addWidget(QLabel("Theme:"), 0, 0)
        layout.addWidget(self.layout_list, 0, 1)
        layout.addWidget(QLabel("ROI render mode:"), 1, 0)
        layout.addWidget(self.labels_render_cmb, 1, 1)
        layout.addWidget(QLabel("Zoom factor for search ROI:"), 2, 0)
        layout.addWidget(self.zoom_factor_spin_box, 2, 1)
        layout.addWidget(QLabel("Show scale bar ticks"), 3, 0)
        layout.addWidget(self.scale_bar_ticks, 3, 1)
        layout.setColumnStretch(2, 1)
        layout.setRowStretch(6, 1)
        self.setLayout(layout)

    def change_scale_bar_ticks(self):
        self.settings.set_in_profile("scale_bar_ticks", self.scale_bar_ticks.isChecked())

    def _update_scale_bar_ticks(self):
        self.scale_bar_ticks.setChecked(self.settings.get_from_profile("scale_bar_ticks", True))

    def change_zoom_factor(self):
        self.settings.set_in_profile(SEARCH_ZOOM_FACTOR_STR, self.zoom_factor_spin_box.value())

    def _update_zoom_factor(self):
        self.zoom_factor_spin_box.setValue(self.settings.get_from_profile(SEARCH_ZOOM_FACTOR_STR))

    def change_theme(self):
        self.settings.theme_name = self.layout_list.currentText()

    def change_render_mode(self, text):
        self.settings.set_in_profile(RENDERING_MODE_NAME_STR, text)

    def _update_render_mode(self):
        self.labels_render_cmb.setCurrentText(
            self.settings.get_from_profile(RENDERING_MODE_NAME_STR, RENDERING_LIST[0])
        )


class ColorControl(QTabWidget):
    """
    Class for storage all settings for labels and colormaps.
    """

    def __init__(self, settings: BaseSettings, image_view_names: List[str]):
        super().__init__()
        self.appearance = Appearance(settings)
        self.colormap_editor = PColormapCreator(settings)
        self.color_preview = PColormapList(settings, image_view_names)
        self.color_preview.edit_signal.connect(self.colormap_editor.set_colormap)
        self.color_preview.edit_signal.connect(self._set_colormap_editor)
        self.label_editor = LabelEditor(settings)
        self.label_view = LabelChoose(settings)
        self.label_view.edit_with_name_signal.connect(self.label_editor.set_colors)
        self.label_view.edit_signal.connect(self._set_label_editor)
        self.mask_control = MaskControl(settings)
        self.addTab(self.appearance, "Appearance")
        self.addTab(self.color_preview, "Color maps")
        self.addTab(self.colormap_editor, "Color Map creator")
        self.addTab(self.label_view, "Select labels")
        self.addTab(self.label_editor, "Create labels")
        self.addTab(self.mask_control, "Mask marking")

    def _set_label_editor(self):
        self.setCurrentWidget(self.label_editor)

    def _set_colormap_editor(self):
        self.setCurrentWidget(self.colormap_editor)


class AdvancedWindow(QTabWidget):
    """
    Base class for advanced windows.
    It contains colormap management connected tabs: :py:class:`.PColormapCreator`
    and :py:class:`.PColormapList`.

    :param settings: program settings
    :param image_view_names: passed as second argument to :py:class:`~.PColormapList`
    """

    def __init__(self, settings: BaseSettings, image_view_names: List[str], reload_list=None, parent=None):
        super().__init__(parent)
        self.color_control = ColorControl(settings, image_view_names)
        self.settings = settings
        self.reload_list = reload_list if reload_list is not None else []

        self.develop = DevelopTab()
        self.addTab(self.color_control, "Color control")
        if state_store.develop:
            self.addTab(self.develop, "Develop")
        if self.window() == self:
            with suppress(KeyError):
                geometry = self.settings.get_from_profile("advanced_window_geometry")
                self.restoreGeometry(QByteArray.fromHex(bytes(geometry, "ascii")))

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Save geometry if widget is used as standalone window.
        """
        if self.window() == self:
            self.settings.set_in_profile("advanced_window_geometry", self.saveGeometry().toHex().data().decode("ascii"))
        super().closeEvent(event)


class ImageMetadata(QWidget):
    def __init__(self, settings: PartSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self._dict_viewer = DictViewer(self.settings.image.metadata if self.settings.image else None)
        self.channel_info = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self._dict_viewer)
        layout.addWidget(self.channel_info)
        self.setLayout(layout)
        self.settings.image_changed.connect(self.update_metadata)

    def update_metadata(self):
        self._dict_viewer.set_data(self.settings.image.metadata)
        self.channel_info.setText(f"Channels: {self.settings.image.channel_names}")
