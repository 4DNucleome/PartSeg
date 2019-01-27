import sys
import os
import appdirs
import platform

app_name = "PartSeg"
app_lab = "LFSG"
config_folder = appdirs.user_data_dir(app_name, app_lab)

if getattr(sys, 'frozen', False):
    os.environ["MPLCONFIGDIR"] = os.path.join(config_folder, ".matplotlib")
    if platform.system() == "Linux":
        os.environ["FONTCONFIG_FILE"] = "/etc/fonts/fonts.conf"
        os.environ["QT_XKB_CONFIG_ROOT"]="/usr/share/X11/xkb"


develop = False

def set_develop(value):
    global develop
    develop = value

src_file_folder = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
static_file_folder = os.path.join(src_file_folder, 'static_files')

big_font_size = 15
button_margin = 10
button_height = 30
button_small_dist = -2


if platform.system() == "Linux":
    big_font_size = 14

if platform.system() == "Darwin":
    big_font_size = 20
    button_margin = 30
    button_height = 34
    button_small_dist = -10

if platform.system() == "Windows":
    big_font_size = 12