import pytest
import typer

from inference_cli.lib.roboflow_cloud.batch_processing.core import parse_key_value


def test_parse_key_value_parses_simple_key_value_pairs() -> None:
    assert parse_key_value(["a=1", "b=2"]) == {"a": "1", "b": "2"}


def test_parse_key_value_returns_empty_dict_for_empty_input() -> None:
    assert parse_key_value([]) == {}


def test_parse_key_value_preserves_equals_signs_inside_value() -> None:
    # split-once semantics — important for query strings, JWTs, paths with `=`, etc.
    assert parse_key_value(["url=https://x?a=b&c=d"]) == {
        "url": "https://x?a=b&c=d"
    }


def test_parse_key_value_raises_bad_parameter_when_separator_missing() -> None:
    with pytest.raises(typer.BadParameter):
        parse_key_value(["no-equals-sign"])


def test_parse_key_value_last_value_wins_on_duplicate_key() -> None:
    # Caller might accidentally repeat the CLI flag for the same workflow input;
    # the dict-overwrite behavior is the observable contract.
    assert parse_key_value(["a=1", "a=2"]) == {"a": "2"}


def test_parse_key_value_allows_empty_value() -> None:
    assert parse_key_value(["k="]) == {"k": ""}
