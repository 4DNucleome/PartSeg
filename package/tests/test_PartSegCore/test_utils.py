from PartSegCore.utils import CallbackFun, CallbackMethod, get_callback


def test_callback_fun():
    call_list = []

    def call_fun():
        call_list.append(1)

    callback = CallbackFun(call_fun)
    assert call_list == []
    callback()
    assert call_list == [1]
    assert callback.is_alive()


def test_callback_method():
    call_list = []

    class A:
        def fun(self):  # pylint: disable=R0201
            call_list.append(1)

    a = A()
    callback = CallbackMethod(a.fun)
    assert call_list == []
    callback()
    assert call_list == [1]
    assert callback.is_alive()
    del a  # skipcq: PTC-W0043
    assert not callback.is_alive()
    callback()
    assert call_list == [1]


def test_get_callback():
    def call_fun():
        return 1

    class A:
        def fun(self):  # pylint: disable=R0201
            return 1

    a = A()

    assert isinstance(get_callback(call_fun), CallbackFun)
    assert isinstance(get_callback(a.fun), CallbackMethod)
