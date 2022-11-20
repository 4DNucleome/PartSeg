"""
Module with default values of application state
"""
import os
import sys

import appdirs
import packaging.version

from PartSeg import APP_LAB, APP_NAME, __version__, parsed_version

always_report: bool = False
#: if report errors using sentry without asking user about it
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
sentry_url = os.environ.get("PARTSEG_SENTRY_URL", "https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")
auto_report = parsed_version.is_devrelease and getattr(sys, "frozen", False)

__all__ = (
    "report_errors",
    "show_error_dialog",
    "custom_plugin_load",
    "check_for_updates",
    "develop",
    "save_suffix",
    "save_folder",
)
