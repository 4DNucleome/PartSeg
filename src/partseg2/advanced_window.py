from PyQt5.QtCore import QByteArray
from PyQt5.QtWidgets import QTabWidget, QWidget
from common_gui.colors_choose import ColorSelector


class AdvancedSettings(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings


class StatisticsWindow(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings


class StatisticsSettings(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings


class AdvancedWindow(QTabWidget):
    """
    :type settings: Settings
    """
    def __init__(self, settings, parent=None):
        super(AdvancedWindow, self).__init__(parent)
        self.settings = settings
        self.setWindowTitle("Settings and statistics")
        self.advanced_settings = AdvancedSettings(settings)
        self.colormap_settings = ColorSelector(settings, ["raw_control", "result_control"])
        self.statistics = StatisticsWindow(settings)
        self.statistics_settings = StatisticsSettings(settings)
        self.addTab(self.advanced_settings, "Settings")
        self.addTab(self.colormap_settings, "Color maps")
        self.addTab(self.statistics, "Statistics")
        self.addTab(self.statistics_settings, "Statistic settings")
        """if settings.advanced_menu_geometry is not None:
            self.restoreGeometry(settings.advanced_menu_geometry)"""
        try:
            geometry = self.settings.get_from_profile("advanced_window_geometry")
            self.restoreGeometry(QByteArray.fromHex(bytes(geometry, 'ascii')))
        except KeyError:
            pass

    def closeEvent(self, *args, **kwargs):
        self.settings.set_in_profile("advanced_window_geometry", bytes(self.saveGeometry().toHex()).decode('ascii'))
        super(AdvancedWindow, self).closeEvent(*args, **kwargs)

