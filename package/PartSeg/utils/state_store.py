import os

import appdirs

from PartSeg import app_name, app_lab, __version__

report_errors = False
show_error_dialog = False
custom_plugin_load = False
check_for_updates = True
develop = False
save_suffix = ""
save_folder = os.path.join(appdirs.user_data_dir(app_name, app_lab), __version__)
