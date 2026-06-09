import pytest
from pydantic import BaseModel

from PartSegImage import Channel


def test_channel_from_channel():
    ch1 = Channel(1)
    ch2 = Channel(2)
    ch3 = Channel(ch1)
    assert ch1 == ch3
    assert ch1 != ch2


def test_representation():
    ch1 = Channel(1)
    assert str(ch1) == "2"
    assert repr(ch1) == "<PartSegImage.channel_class.Channel(value=1)>"


def test_as_dict():
    ch1 = Channel(1)
    assert ch1.as_dict() == {"value": 1}
    ch2 = Channel(**ch1.as_dict())
    assert ch1 == ch2


class Model(BaseModel):
    channel: Channel


MODEL_SCHEMA = {
    "properties": {
        "channel": {
            "anyOf": [{"type": "integer"}, {"type": "string", "minLength": 1}],
            "description": "Image channel index or channel name. Accepts an integer or any non-empty string.",
            "examples": [0, "nucleus"],
            "title": "Channel",
        }
    },
    "required": ["channel"],
    "title": "Model",
    "type": "object",
}


@pytest.mark.parametrize("value", [Channel(1), 1, Channel("1"), "1"], ids=lambda x: f"value={x!r}")
def test_as_pydantic_field(value):
    model = Model(channel=value)
    assert isinstance(model.channel, Channel)
    assert model.channel == Channel(value)


def test_json_schema():
    assert Model.model_json_schema() == MODEL_SCHEMA
