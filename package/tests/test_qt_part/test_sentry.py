import pytest
import sentry_sdk
import sentry_sdk.utils
from sentry_sdk.hub import Hub
from sentry_sdk.client import Client
from sentry_sdk.serializer import serialize


def test_message_clip(monkeypatch):
    message = "a" * 5000
    assert len(sentry_sdk.utils.strip_string(message).value) == 512
    monkeypatch.setattr(sentry_sdk.utils, "MAX_STRING_LENGTH", 10**4)
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
            event_id = sentry_sdk.capture_event(event, hint=hint)
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
        assert happen[0] is True
