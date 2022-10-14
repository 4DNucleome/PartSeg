"""
Module with default values of application state
"""
import os

import appdirs
import packaging.version

from PartSeg import APP_LAB, APP_NAME, __version__

report_errors = False
show_error_dialog = False
custom_plugin_load = False
check_for_updates = True
develop = False
save_suffix = ""
save_folder = os.path.join(
    os.environ.get("PARTSEG_SETTINGS_DIR", appdirs.user_data_dir(APP_NAME, APP_LAB)),
    str(packaging.version.parse(__version__).base_version),
)

__all__ = (
    "report_errors",
    "show_error_dialog",
    "custom_plugin_load",
    "check_for_updates",
    "develop",
    "save_suffix",
    "save_folder",
)
