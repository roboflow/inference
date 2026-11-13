import pytest

from inference_server.gateway import is_loaded_model_entry, loaded_model_entries


@pytest.mark.parametrize(
    "entry, expected",
    [
        ({"loaded": True}, True),
        ({"loaded": False, "loading": True}, True),
        ({"loaded": False, "loading": False}, False),
        ({"loaded": False, "loading": False, "state": "unloaded"}, False),
        ({"state": "loaded"}, True),
        ({"state": "loading"}, True),
        ({"state": "draining"}, True),
        ({"state": "unhealthy"}, True),
        ({"state": "not_loaded"}, False),
        ({"state": "unloaded"}, False),
    ],
)
def test_loaded_model_entries_applies_the_loaded_rule(entry, expected):
    assert bool(loaded_model_entries({"m/1": entry})) is expected


def test_loaded_model_entries_preserves_order():
    models = {
        "c/1": {"loaded": True},
        "a/1": {"loaded": False},
        "b/1": {"state": "loaded"},
    }

    assert list(loaded_model_entries(models)) == ["c/1", "b/1"]


def test_non_dict_entry_is_not_loaded():
    assert is_loaded_model_entry("coco/3") is False
    assert loaded_model_entries({"coco/3": "garbage", "ok/1": {"state": "loaded"}}) == {
        "ok/1": {"state": "loaded"}
    }
