import multiprocessing
from unittest.mock import Mock

import numpy as np
import pytest
import sentry_sdk
import sentry_sdk.serializer
import sentry_sdk.utils
from sentry_sdk.client import Client
from sentry_sdk.serializer import serialize

from PartSegCore.utils import prepare_error_data, safe_repr

BASE_SIZE = 10**4
NEW_CLIP_LIMIT = BASE_SIZE * 10
TEST_SIZE_DATA = BASE_SIZE + 10


def test_sentry_variables_clip(monkeypatch):
    a = b = c = d = e = f = g = h = i = j = k = m = n = o = p = r = s = t = 1  # noqa: F841
    try:
        raise ValueError("eeee")
    except ValueError as ee:
        event, _hint = sentry_sdk.utils.event_from_exception(ee)
        clipped = serialize(event)
        assert (
            len(clipped["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"])
            == sentry_sdk.serializer.MAX_DATABAG_BREADTH
        )


def test_sentry_variables_clip_change_breadth(monkeypatch):
    monkeypatch.setattr(sentry_sdk.serializer, "MAX_DATABAG_BREADTH", 100)
    letters = "abcdefghijkmnoprst"
    a = b = c = d = e = f = g = h = i = j = k = m = n = o = p = r = s = t = 1  # noqa: F841
    try:
        raise ValueError("eeee")
    except ValueError as ee:
        event, hint = sentry_sdk.utils.event_from_exception(ee)
        vars_dict = event["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]
        for letter in letters:
            assert letter in vars_dict

        clipped = serialize(event)
        assert len(clipped["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]) == len(vars_dict)
        assert len(clipped["exception"]["values"][0]["stacktrace"]["frames"][0]["vars"]) > 10
        client = Client("https://aaa@test.pl/77")
        with sentry_sdk.new_scope() as scope:
            scope.set_client(client)
            sentry_sdk.capture_event(event, hint=hint)


def test_sentry_report(monkeypatch):
    message = "a" * TEST_SIZE_DATA
    mock = Mock()

    def check_event(event):
        mock(event)
        assert len(event["message"]) == TEST_SIZE_DATA
        assert len(event["extra"]["lorem"]) == TEST_SIZE_DATA

    def check_envelope(envelope):
        check_event(envelope.get_event())

    try:
        raise ValueError("eeee")
    except ValueError as e:
        event, hint = sentry_sdk.utils.event_from_exception(e)
        event["message"] = message
        client = Client("https://aaa@test.pl/77")
        monkeypatch.setattr(client.transport, "capture_envelope", check_envelope)

        with sentry_sdk.new_scope() as scope:
            scope.set_client(client)
            scope.set_extra("lorem", message)
            sentry_sdk.capture_event(event, hint=hint)
    mock.assert_called_once()


def test_sentry_report_no_clip(monkeypatch):
    message = "a" * 5000
    mock = Mock()

    def check_event(event):
        mock(event)
        assert len(event["message"]) == 5000
        assert len(event["extra"]["lorem"]) == 5000

    def check_envelope(envelope):
        check_event(envelope.get_event())

    try:
        raise ValueError("eeee")
    except ValueError as e:
        event, hint = sentry_sdk.utils.event_from_exception(e)
        event["message"] = message
        client = Client("https://aaa@test.pl/77", max_value_length=10**4)
        monkeypatch.setattr(client.transport, "capture_envelope", check_envelope)
        with sentry_sdk.new_scope() as scope:
            scope.set_client(client)
            scope.set_extra("lorem", message)
            event_id = sentry_sdk.capture_event(event, hint=hint)
        assert event_id is not None
    mock.assert_called_once()


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
    done = [False]

    def check_event(envelope):
        assert len(envelope.items[0].payload.json["exception"]["values"][0]["stacktrace"]["frames"]) == 12
        done[0] = True

    message_queue = multiprocessing.get_context("spawn").Queue()
    p = multiprocessing.get_context("spawn").Process(target=executor_fun, args=(message_queue,))
    p.start()
    p.join()
    assert not message_queue.empty()
    ex, event, _tr = message_queue.get()
    assert isinstance(ex, ValueError)
    assert isinstance(event, dict)
    client = Client("https://aaa@test.pl/77")
    monkeypatch.setattr(client.transport, "capture_envelope", check_event)
    with sentry_sdk.new_scope() as scope:
        scope.set_client(client)
        event_id = sentry_sdk.capture_event(event)

    assert event_id is not None
    assert done[0]


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
