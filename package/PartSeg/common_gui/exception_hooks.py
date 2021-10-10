import logging

from PartSeg.common_backend.except_hook import show_warning

OPEN_ERROR = "Open error"


def load_data_exception_hook(exception):
    if isinstance(exception, ValueError) and exception.args[0] == "Incompatible shape of mask and image":
        show_warning(OPEN_ERROR, "Most probably you try to load mask from other image. Check selected files.")
    elif isinstance(exception, MemoryError):
        show_warning(OPEN_ERROR, f"Not enough memory to read this image: {exception}")
    elif isinstance(exception, IOError):
        show_warning(OPEN_ERROR, f"Some problem with reading from disc: {exception}")
    elif isinstance(exception, KeyError):
        show_warning(OPEN_ERROR, f"Some problem project file: {exception}")
        logging.warning(exception)
    else:
        raise exception
