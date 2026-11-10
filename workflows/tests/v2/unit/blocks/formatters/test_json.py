"""``roboflow_core/json_parser@v1`` on plain strings: V1 parsing and configuration rules."""

import pytest
from pydantic import ValidationError
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock, output_json_keys, parse_json_fields
from roboflow_workflows.v2.data.kinds import BOOLEAN_KIND, WILDCARD_KIND


def _run(raw_json: str, expected_fields) -> dict:
    params = JsonParserBlock.Params(raw_json=raw_json, expected_fields=expected_fields)
    result = JsonParserBlock().run(
        raw_json=params.raw_json, expected_fields=params.expected_fields
    )

    return result


def test_object_fields_are_extracted_in_declared_order() -> None:
    # when
    result = _run('{"confidence": 0.8, "model_id": "m"}', ["model_id", "confidence"])

    # then
    assert result == {"model_id": "m", "confidence": 0.8, "error_status": False}
    assert list(result) == ["model_id", "confidence", "error_status"]


def test_missing_field_is_none_and_sets_error_status() -> None:
    # when
    result = _run('{"a": 1}', ["a", "b"])

    # then
    assert result == {"a": 1, "b": None, "error_status": True}


def test_present_null_counts_as_found() -> None:
    # when
    result = _run('{"a": null}', ["a"])

    # then
    assert result == {"a": None, "error_status": False}


def test_values_keep_their_json_types() -> None:
    # when
    result = _run(
        '{"n": 3, "f": 0.5, "b": true, "s": "x", "l": [1, {"k": 2}], "o": {"k": [3]}}',
        ["n", "f", "b", "s", "l", "o"],
    )

    # then
    assert result == {
        "n": 3,
        "f": 0.5,
        "b": True,
        "s": "x",
        "l": [1, {"k": 2}],
        "o": {"k": [3]},
        "error_status": False,
    }


@pytest.mark.parametrize("raw_json", ["not json", "", "{'a': 1}", '{"a": 1'])
def test_invalid_json_gives_none_fields_and_error(raw_json: str) -> None:
    # when
    result = _run(raw_json, ["a"])

    # then
    assert result == {"a": None, "error_status": True}


@pytest.mark.parametrize("raw_json", ["not json", ""])
def test_invalid_json_without_fields_still_reports_error(raw_json: str) -> None:
    # when
    result = _run(raw_json, [])

    # then
    assert result == {"error_status": True}


def test_too_deep_nesting_is_a_parse_error() -> None:
    # given: deeper than the JSON decoder's recursion limit
    raw_json = "[" * 100_000 + "]" * 100_000

    # when
    result = _run(raw_json, ["a"])

    # then
    assert result == {"a": None, "error_status": True}


@pytest.mark.parametrize("raw_json", ["[1, 2]", '"abc"', "3", '["a"]', "null"])
def test_non_object_json_with_fields_is_an_error(raw_json: str) -> None:
    # when
    result = _run(raw_json, ["a"])

    # then
    assert result == {"a": None, "error_status": True}


@pytest.mark.parametrize("raw_json", ["[1, 2]", "3", "null"])
def test_non_object_json_without_fields_is_not_an_error(raw_json: str) -> None:
    # when
    result = _run(raw_json, [])

    # then
    assert result == {"error_status": False}


def test_last_duplicate_json_key_wins_and_nan_is_accepted() -> None:
    # when
    result = _run('{"a": 1, "a": 2, "b": NaN}', ["a", "b"])

    # then
    assert result["a"] == 2
    assert result["b"] != result["b"]
    assert result["error_status"] is False


def test_dotted_key_is_a_literal_key_not_a_path() -> None:
    # when
    result = _run('{"a": {"b": 1}, "a.b": 2}', {"nested": "a.b"})

    # then
    assert result == {"nested": 2, "error_status": False}


def test_only_the_first_markdown_fence_is_read() -> None:
    # given: the tag is case-insensitive; text around the fences is ignored
    raw_json = 'Sure!\n```JSON\n{"a": 1}\n```\nand\n```json\n{"a": 2}\n```'

    # when
    result = _run(raw_json, ["a"])

    # then
    assert result == {"a": 1, "error_status": False}


def test_invalid_first_fence_does_not_fall_back_to_a_later_one() -> None:
    # when
    result = _run('```json\nbroken\n```\n```json\n{"a": 2}\n```', ["a"])

    # then
    assert result == {"a": None, "error_status": True}


@pytest.mark.parametrize(
    "raw_json",
    [
        '```\n{"a": 1}\n```',
        '```jsonc\n{"a": 1}\n```',
        '```json\n{"a": 1}',
    ],
    ids=["untagged-fence", "jsonc-tag", "unclosed-fence"],
)
def test_fences_v1_does_not_recognise_are_parse_errors(raw_json: str) -> None:
    # when
    result = _run(raw_json, ["a"])

    # then
    assert result == {"a": None, "error_status": True}


def test_surrounding_whitespace_is_allowed() -> None:
    # when
    result = _run('  \n{"a": 1}\n  ', ["a"])

    # then
    assert result == {"a": 1, "error_status": False}


def test_mapping_reads_keys_a_selector_cannot_name() -> None:
    # given
    expected_fields = {
        "label": "class name",
        "upstream_error": "error_status",
        "empty_key": "",
    }

    # when
    result = _run(
        '{"class name": "car", "error_status": "timeout", "": 7}', expected_fields
    )

    # then
    assert result == {
        "label": "car",
        "upstream_error": "timeout",
        "empty_key": 7,
        "error_status": False,
    }


def test_two_outputs_may_read_the_same_key() -> None:
    # when
    result = _run('{"k": 1}', {"first": "k", "second": "k"})

    # then
    assert result == {"first": 1, "second": 1, "error_status": False}


def test_mapping_missing_key_sets_error_status() -> None:
    # when
    result = _run('{"class name": "car"}', {"label": "class name", "score": "score"})

    # then
    assert result == {"label": "car", "score": None, "error_status": True}


def test_parse_helper_returns_status_and_values() -> None:
    # when
    status, values = parse_json_fields('{"x": 1}', json_keys={"out": "x"})

    # then
    assert status is False
    assert values == {"out": 1}


def test_output_json_keys_for_both_forms() -> None:
    assert output_json_keys(["a", "b"]) == {"a": "a", "b": "b"}
    assert output_json_keys({"out": "in key"}) == {"out": "in key"}


def test_describe_outputs_lists_fields_then_error_status() -> None:
    # given
    params = JsonParserBlock.Params(
        raw_json="{}", expected_fields={"label": "class name", "score": "score"}
    )

    # when
    outputs = JsonParserBlock.describe_outputs(params)

    # then
    assert list(outputs) == ["label", "score", "error_status"]
    assert outputs["label"].kinds == (WILDCARD_KIND,)
    assert outputs["error_status"].kinds == (BOOLEAN_KIND,)


@pytest.mark.parametrize("expected_fields", [[], {}])
def test_no_fields_leaves_only_error_status(expected_fields) -> None:
    # given
    params = JsonParserBlock.Params(raw_json="{}", expected_fields=expected_fields)

    # when
    outputs = JsonParserBlock.describe_outputs(params)

    # then
    assert list(outputs) == ["error_status"]


@pytest.mark.parametrize(
    "expected_fields, message",
    [
        (["error_status"], "'error_status' is reserved"),
        ({"error_status": "x"}, "'error_status' is reserved"),
        (["class name"], "{'class_name': 'class name'}"),
        (["a.b"], "{'a_b': 'a.b'}"),
        ([""], "{'field': ''}"),
        ({"bad name": "x"}, "output name 'bad name' must use letters"),
        ({"": "x"}, "output name '' must use letters"),
        (["a", "b", "a"], "duplicate expected_fields entries: ['a']"),
    ],
)
def test_invalid_output_names_are_rejected(expected_fields, message: str) -> None:
    # when
    with pytest.raises(ValidationError) as error:
        JsonParserBlock.Params(raw_json="{}", expected_fields=expected_fields)

    # then
    assert message in str(error.value)


@pytest.mark.parametrize(
    "expected_fields",
    ["a", {"a": 1}, {"a": None}, {"a": ["k"]}, [1], None],
    ids=["string", "int-key", "none-key", "list-key", "int-entry", "none"],
)
def test_malformed_expected_fields_are_rejected(expected_fields) -> None:
    with pytest.raises(ValidationError):
        JsonParserBlock.Params(raw_json="{}", expected_fields=expected_fields)
