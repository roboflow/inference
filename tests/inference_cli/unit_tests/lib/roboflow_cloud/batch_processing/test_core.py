import pytest
import typer

from inference_cli.lib.roboflow_cloud.batch_processing.core import parse_key_value


def test_parse_key_value_parses_simple_key_value_pairs() -> None:
    assert parse_key_value(["a=1", "b=2"]) == {"a": "1", "b": "2"}


def test_parse_key_value_returns_empty_dict_for_empty_input() -> None:
    assert parse_key_value([]) == {}


def test_parse_key_value_preserves_equals_signs_inside_value() -> None:
    assert parse_key_value(["url=https://x?a=b&c=d"]) == {
        "url": "https://x?a=b&c=d"
    }


def test_parse_key_value_raises_bad_parameter_when_separator_missing() -> None:
    with pytest.raises(typer.BadParameter):
        parse_key_value(["no-equals-sign"])


def test_parse_key_value_last_value_wins_on_duplicate_key() -> None:
    assert parse_key_value(["a=1", "a=2"]) == {"a": "2"}


def test_parse_key_value_prevents_empty_values_by_default() -> None:
    with pytest.raises(typer.BadParameter):
        assert parse_key_value(["k="])


def test_parse_key_value_prevents_empty_keys_by_default() -> None:
    with pytest.raises(typer.BadParameter):
        assert parse_key_value(["=v"])
