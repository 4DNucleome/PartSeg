import importlib
import sys
from functools import partial
from typing import List

from qtpy.QtGui import QCloseEvent
from qtpy.QtCore import QByteArray
from qtpy.QtWidgets import QTabWidget, QWidget, QPushButton, QGridLayout

from PartSeg import plugins
from PartSeg.common_gui.colormap_creator import PColormapCreator, PColormapList
from PartSeg.common_backend.base_settings import ViewSettings
from PartSeg.utils import register, state_store

"""
This module contains base for advanced window for PartSeg. 
In this moment controlling colormaps tabs and developer utils 
"""


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
        self.settings = settings
        self.colormap_selector = PColormapCreator(settings)
        self.color_preview = PColormapList(settings, image_view_names)
        self.color_preview.edit_signal.connect(self.colormap_selector.set_colormap)
        self.color_preview.edit_signal.connect(partial(self.setCurrentWidget, self.colormap_selector))
        self.develop = DevelopTab()
        self.addTab(self.color_preview, "Color maps")
        self.addTab(self.colormap_selector, "Color Map creator")
        if state_store.develop:
            self.addTab(self.develop, "Develop")
        if self.window() == self:
            try:
                geometry = self.settings.get_from_profile("advanced_window_geometry")
                self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
            except KeyError:
                pass

    def closeEvent(self, event: QCloseEvent) -> None:
        """
        Save geometry if widget is used as standalone window.
        """
        if self.window() == self:
            self.settings.set_in_profile("advanced_window_geometry", self.saveGeometry().toHex().data().decode('ascii'))
        super(AdvancedWindow, self).closeEvent(event)
