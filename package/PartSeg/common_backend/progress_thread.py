import inspect
from typing import Union

from qtpy.QtCore import QThread, Signal


class ProgressTread(QThread):
    range_changed = Signal(int, int)
    step_changed = Signal(int)
    error_signal = Signal(Exception)

    def info_function(self, label: str, val: int):
        if label == "max":
            self.range_changed.emit(0, val)
        elif label == "step":
            self.step_changed.emit(val)


class ExecuteFunctionThread(ProgressTread):
    """Generic Thread to execute """

    def __init__(self, fun, args: Union[list, tuple] = None, kwargs: dict = None):
        super().__init__()
        self.args = args or []
        self.kwargs = kwargs or {}
        self.function = fun
        self.result = None

    def run(self):
        try:
            if "callback_function" in inspect.signature(self.function).parameters:
                self.result = self.function(*self.args, **self.kwargs, callback_function=self.info_function)
            elif (
                "range_changed" in inspect.signature(self.function).parameters
                and "step_changed" in inspect.signature(self.function).parameters
            ):
                self.result = self.function(
                    *self.args,
                    **self.kwargs,
                    range_changed=self.range_changed.emit,
                    step_changed=self.step_changed.emit,
                )
            else:
                self.result = self.function(*self.args, **self.kwargs)

        except Exception as e:
            self.error_signal.emit(e)
