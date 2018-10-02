import abc

from PyQt5.QtCore import QObject


class QtMeta(type(QObject), abc.ABCMeta):
    pass