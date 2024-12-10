import os

from magicgui.widgets import Container, create_widget
from napari.layers import Layer

from PartSeg._roi_analysis.partseg_settings import PartSettings
from PartSeg.common_backend import napari_get_settings
from PartSegCore import Units
from PartSegCore.json_hooks import PartSegEncoder

_SETTINGS = None


class PartSegNapariEncoder(PartSegEncoder):
    def default(self, o):
        if isinstance(o, Layer):
            return o.name
        return super().default(o)


class PartSegNapariSettings(PartSettings):
    json_encoder_class = PartSegNapariEncoder

    @property
    def io_units(self) -> Units:
        return self.get("io_units", Units.nm)

    @io_units.setter
    def io_units(self, value: Units):
        self.set("io_units", value)


def get_settings() -> PartSegNapariSettings:
    global _SETTINGS  # noqa: PLW0603  # pylint: disable=global-statement
    if _SETTINGS is None:
        napari_settings = napari_get_settings()
        if hasattr(napari_settings, "path"):
            save_path = napari_settings.path
        else:
            save_path = os.path.dirname(napari_settings.config_path)
        _SETTINGS = PartSegNapariSettings(os.path.join(save_path, "PartSeg_napari_plugins"))
        _SETTINGS.load()
    return _SETTINGS


class SettingsEditor(Container):
    def __init__(self):
        super().__init__()
        self.settings = get_settings()
        self.units_select = create_widget(self.settings.io_units, annotation=Units, label="Units for io")
        self.units_select.changed.connect(self.units_selection_changed)
        self.settings.connect_("io_units", self.units_changed)
        self.append(self.units_select)

    def units_selection_changed(self, value):
        self.settings.io_units = value
        self.settings.dump()

    def units_changed(self):
        self.units_select.value = self.settings.io_units
