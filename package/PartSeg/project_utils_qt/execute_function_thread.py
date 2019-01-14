from .progress_thread import ProgressTread
from typing import Union


class ExecuteFunctionThread(ProgressTread):
    def __init__(self, fun, args: Union[list, tuple] = None, kwargs: dict = None):
        super().__init__()
        self.args = args if args else []
        self.kwargs = kwargs if kwargs else {}
        self.function = fun
        self.result = None

    def run(self):
        try:
            self.result = self.function(*self.args, **self.kwargs, range_changed=self.range_changed.emit,
                                        step_changed=self.step_changed.emit)
        except Exception as e:
            self.error_signal.emit(e)
