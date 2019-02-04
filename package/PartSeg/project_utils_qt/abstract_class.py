import abc

from qtpy.QtCore import QObject


class QtMeta(type(QObject), abc.ABCMeta):
    pass