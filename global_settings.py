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

if sys.version_info.major == 2:
    import pkgutil
    loader = pkgutil.find_loader("PyQt5")
    if loader is not None:
        use_qt5 = True
    else:
        use_qt5 = False
else:
    import importlib
    spam_spec = importlib.find_loader("PyQt5")
    if spam_spec is not None:
        use_qt5 = True
    else:
        use_qt5 = False

develop = False


def set_qt4():
    global use_qt5
    use_qt5 = False


def set_qt5():
    global use_qt5
    use_qt5 = True


def set_develop(value):
    global develop
    develop = value


file_folder = os.path.dirname(os.path.realpath(__file__))

