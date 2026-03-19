from enum import Enum

from local_migrator import register_class


# noinspection NonAsciiCharacters
@register_class(old_paths=["PartSeg.utils.universal_const.Units"])
class Units(Enum):
    mm = 0
    µm = 1  # noqa: PLC2401
    nm = 2
    pm = 3

    def __str__(self):
        return self.name.replace("_", " ")


_UNITS_LIST = ["mm", "µm", "nm", "pm"]
UNIT_SCALE = [10.0**3, 10.0**6, 10.0**9, 10.0**12]


@register_class()
class LayerNamingFormat(Enum):
    channel_only = 0
    filename_only = 1
    filename_channel = 2
    channel_filename = 3

    def __str__(self):
        return self.name.replace("_", " ")


def format_layer_name(layer_format: LayerNamingFormat, file_name: str, channel_name: str) -> str:
    if layer_format == LayerNamingFormat.channel_only:
        return channel_name
    if layer_format == LayerNamingFormat.filename_only:
        return file_name
    if layer_format == LayerNamingFormat.filename_channel:
        return f"{file_name} | {channel_name}"
    if layer_format == LayerNamingFormat.channel_filename:
        return f"{channel_name} | {file_name}"
    raise ValueError("Unknown format")  # pragma: no cover
