from enum import Enum


class CallbackState(Enum):
    emmit = 0
    stack = 1
    skip = 2


class HierarchicalCallback(object):
    def __init__(self):
        super(HierarchicalCallback, self).__init__()
        self._position_list = set()
        self._state = CallbackState.emmit

    def register_callback(self, name):
        self._position_list.add(name)

    def block_signal(self):
        self._state = CallbackState.skip

    def bound_callback(self, callback_list):
        """
        :type callback_list: list[(() -> (), int)]
        :param callback_list:
        :return:
        """

    def emit(self, name):
        if name not in self._position_list:
            raise ValueError("Signal {} do not known".format(name))
        if self._state == CallbackState.skip:
            return
        elif self._state == CallbackState.stack:
            pass
        else:
            pass