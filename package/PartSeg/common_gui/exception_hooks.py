import sys

from qtpy.QtCore import QMetaObject, Qt
from qtpy.QtWidgets import QApplication

from PartSegCore.io_utils import WrongFileTypeException


def load_data_exception_hook(exception):
    instance = QApplication.instance()
    if isinstance(exception, ValueError) and exception.args[0] == "Incompatible shape of mask and image":
        instance.warning = (
            "Open error",
            "Most probably you try to load mask from other image. Check selected files",
        )
        QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
    elif isinstance(exception, MemoryError):
        instance.warning = "Open error", f"Not enough memory to read this image: {exception}"
        QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
    elif isinstance(exception, IOError):
        instance.warning = "Open error", f"Some problem with reading from disc: {exception}"
        QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
    elif isinstance(exception, KeyError):
        instance.warning = "Open error", f"Some problem project file: {exception}"
        QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
        print(exception, file=sys.stderr)
    elif isinstance(exception, WrongFileTypeException):
        instance.warning = (
            "Open error",
            "No needed files inside archive. Most probably you choose file from segmentation mask",
        )
        QMetaObject.invokeMethod(instance, "show_warning", Qt.QueuedConnection)
    else:
        raise exception
