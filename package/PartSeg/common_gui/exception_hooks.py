import logging

from PartSeg.custom_application import CustomApplication

OPEN_ERROR = "Open error"


def load_data_exception_hook(exception):
    instance: CustomApplication = CustomApplication.instance()
    if isinstance(exception, ValueError) and exception.args[0] == "Incompatible shape of mask and image":
        instance.show_warning(OPEN_ERROR, "Most probably you try to load mask from other image. Check selected files.")
    elif isinstance(exception, MemoryError):
        instance.show_warning(OPEN_ERROR, f"Not enough memory to read this image: {exception}")
    elif isinstance(exception, IOError):
        instance.show_warning(OPEN_ERROR, f"Some problem with reading from disc: {exception}")
    elif isinstance(exception, KeyError):
        instance.show_warning(OPEN_ERROR, f"Some problem project file: {exception}")
        logging.warning(exception)
    else:
        raise exception
