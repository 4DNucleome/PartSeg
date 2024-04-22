"""
Module with default values of application state
"""

import os
import sys

import appdirs
import packaging.version

from PartSeg import APP_LAB, APP_NAME, __version__, parsed_version

#: if true then report errors using sentry without asking user
auto_report = parsed_version.is_devrelease and getattr(sys, "frozen", False)
#: if report errors using sentry without asking user about it
always_report = False
#: if enable error reporting button (to allow see stacktrace but not leak information)
report_errors = False
#: if show error dialog with stacktrace, if not then only print it to console
show_error_dialog = False
#: if load inner plugins (plugins that are not in separate package)
custom_plugin_load = False
#: if check for updates on startup,
check_for_updates = True
#: if run in develop mode (show developer tab in advanced window)
develop = False

save_suffix = ""
#: path to folder where save settings
save_folder = os.path.join(
    os.environ.get("PARTSEG_SETTINGS_DIR", appdirs.user_data_dir(APP_NAME, APP_LAB)),
    str(packaging.version.parse(__version__).base_version),
)

sentry_url = os.environ.get("PARTSEG_SENTRY_URL", "https://d4118280b73d4ee3a0222d0b17637687@sentry.io/1309302")

__all__ = (
    "auto_report",
    "always_report",
    "custom_plugin_load",
    "check_for_updates",
    "develop",
    "report_errors",
    "save_suffix",
    "save_folder",
    "sentry_url",
    "show_error_dialog",
)
