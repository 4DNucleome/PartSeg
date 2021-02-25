import multiprocessing

import numpy as np
import pytest
import sentry_sdk
import sentry_sdk.serializer
import sentry_sdk.utils
from sentry_sdk.client import Client
from sentry_sdk.hub import Hub
from sentry_sdk.serializer import serialize

from PartSeg.common_backend.base_argparser import safe_repr
from PartSegCore.analysis.batch_processing.batch_backend import prepare_error_data


def test_message_clip(monkeypatch):
    message = "a" * 5000
    assert len(sentry_sdk.utils.strip_string(message).value) == 512
    monkeypatch.setattr(sentry_sdk.utils, "MAX_STRING_LENGTH", 10 ** 4)
    assert len(sentry_sdk.utils.strip_string(message)) == 5000


def test_sentry_serialize_clip(monkeypatch):
    message = "a" * 5000
    try:
        raise ValueError("eeee")
    except ValueError as e:
        event, hint = sentry_sdk.utils.event_from_exception(e)
        event["message"] = message

        cliped = serialize(event)
        assert len(cliped["message"]) == 512
        monkeypatch.setattr(sentry_sdk.utils, "MAX_STRING_LENGTH", 10 ** 4)
        cliped = serialize(event)
        assert len(cliped["message"]) == 5000


def test_sentry_report(monkeypatch):
    message = "a" * 5000
    happen = [False]

    def check_event(event):
        happen[0] = True
        assert len(event["message"]) == 512
        assert len(event["extra"]["lorem"]) == 512

    try:
        raise ValueError("eeee")
    except ValueError as e:
        event, hint = sentry_sdk.utils.event_from_exception(e)
        event["message"] = message
        client = Client("https://aaa@test.pl/77")
        Hub.current.bind_client(client)
        monkeypatch.setattr(client.transport, "capture_event", check_event)
        with sentry_sdk.push_scope() as scope:
            scope.set_extra("lorem", message)
            sentry_sdk.capture_event(event, hint=hint)
        assert happen[0] is True


def test_sentry_report_no_clip(monkeypatch):
    message = "a" * 5000
    happen = [False]
    monkeypatch.setattr(sentry_sdk.utils, "MAX_STRING_LENGTH", 10 ** 4)

    def check_event(event):
        happen[0] = True
        assert len(event["message"]) == 5000
        assert len(event["extra"]["lorem"]) == 5000

    try:
        raise ValueError("eeee")
    except ValueError as e:
        event, hint = sentry_sdk.utils.event_from_exception(e)
        event["message"] = message
        client = Client("https://aaa@test.pl/77")
        Hub.current.bind_client(client)
        monkeypatch.setattr(client.transport, "capture_event", check_event)
        with sentry_sdk.push_scope() as scope:
            scope.set_extra("lorem", message)
            event_id = sentry_sdk.capture_event(event, hint=hint)
        assert event_id is not None
        assert happen[0] is True


def exception_fun(num: int):
    if num < 1:
        raise ValueError("test")
    exception_fun(num - 1)


def executor_fun(que: multiprocessing.Queue):
    try:
        exception_fun(10)
    except ValueError as e:
        ex, (event, tr) = prepare_error_data(e)
        que.put((ex, event, tr))


def test_exception_pass(monkeypatch):
    def check_event(event):
        assert len(event["exception"]["values"][0]["stacktrace"]["frames"]) == 12

    message_queue = multiprocessing.get_context("spawn").Queue()
    p = multiprocessing.get_context("spawn").Process(target=executor_fun, args=(message_queue,))
    p.start()
    p.join()
    assert not message_queue.empty()
    ex, event, _tr = message_queue.get()
    assert isinstance(ex, ValueError)
    assert isinstance(event, dict)
    client = Client("https://aaa@test.pl/77")
    Hub.current.bind_client(client)
    monkeypatch.setattr(client.transport, "capture_event", check_event)
    event_id = sentry_sdk.capture_event(event)
    assert event_id is not None


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.float32])
def test_numpy_array_serialize(monkeypatch, dtype):
    arr = np.zeros((10, 10), dtype=dtype)
    arr[1, 5] = 10
    monkeypatch.setattr(sentry_sdk.serializer, "safe_repr", safe_repr)
    res = serialize(arr)
    assert res == f"array(size={arr.size}, shape={arr.shape}, dtype={arr.dtype}, min={arr.min()}, max={arr.max()})"


def test_small_numpy_serialize(monkeypatch):
    arr = np.zeros(10, dtype=np.uint8)
    arr[1] = 10
    monkeypatch.setattr(sentry_sdk.serializer, "safe_repr", safe_repr)
    res = serialize(arr)
    assert res == str(res)
