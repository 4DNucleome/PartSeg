import sys
if sys.version_info.major == 2:
    import pkgutil
    loader = pkgutil.find_loader("PyQt5")
    if loader is not None:
        use_qt5 = True
    else:
        use_qt5 = False
else:
    import importlib
    spam_spec = importlib.util.find_spec("PyQt5")
    if spam_spec is not None:
        use_qt5 = True
    else:
        use_qt5 = False


def set_qt4():
    global use_qt5
    use_qt5 = False


def set_qt5():
    global use_qt5
    use_qt5 = True
