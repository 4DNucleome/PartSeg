from .progress_thread import ProgressTread
from typing import Union
import inspect


class ExecuteFunctionThread(ProgressTread):
    def __init__(self, fun, args: Union[list, tuple] = None, kwargs: dict = None):
        super().__init__()
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}
        self.function = fun
        self.result = None

    def run(self):
        try:
            if "callback_function" in inspect.signature(self.function).parameters:
                self.result = self.function(*self.args, **self.kwargs, callback_function=self.info_function)
            elif "range_changed" in inspect.signature(self.function).parameters and \
                    "step_changed" in inspect.signature(self.function).parameters:
                self.result = self.function(*self.args, **self.kwargs, range_changed=self.range_changed.emit,
                                            step_changed=self.step_changed.emit)
            else:
                self.result = self.function(*self.args, **self.kwargs)

        except Exception as e:
            self.error_signal.emit(e)
