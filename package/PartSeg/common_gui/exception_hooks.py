import logging

from PartSeg.common_backend.except_hook import show_warning
from PartSegImage.image_reader import INCOMPATIBLE_IMAGE_MASK

OPEN_ERROR = "Open error"


def load_data_exception_hook(exception):
    if isinstance(exception, ValueError) and exception.args[0] == INCOMPATIBLE_IMAGE_MASK:
        # TODO think about incompatibilible axes (image_reader:417)
        show_warning(
            OPEN_ERROR,
            "Most probably you try to load mask from other image. Check selected files.",
            exception=exception,
        )
    elif isinstance(exception, MemoryError):
        show_warning(OPEN_ERROR, f"Not enough memory to read this image: {exception}", exception=exception)
    elif isinstance(exception, IOError):
        show_warning(OPEN_ERROR, f"Some problem with reading from disc: {exception}", exception=exception)
    elif isinstance(exception, KeyError):
        show_warning(OPEN_ERROR, f"Some problem project file: {exception}", exception=exception)
        logging.warning(exception)
    else:
        raise exception
