import pytest

from inference.core.workflows.core_steps.common.vlm_json import (
    coerce_classification_payload,
    extract_json_payload,
)


def test_extract_json_payload_when_plain_json_object_given() -> None:
    # when
    error_status, payload = extract_json_payload('{"a": 1}')

    # then
    assert error_status is False
    assert payload == {"a": 1}


def test_extract_json_payload_when_plain_json_list_given() -> None:
    # when
    error_status, payload = extract_json_payload('[{"a": 1}, {"a": 2}]')

    # then
    assert error_status is False
    assert payload == [{"a": 1}, {"a": 2}]


def test_extract_json_payload_prefers_first_json_markdown_block() -> None:
    # given
    raw = '```json\n{"first": true}\n```\n```json\n{"second": true}\n```'

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == {"first": True}


def test_extract_json_payload_when_fence_has_other_language_tag() -> None:
    # when
    error_status, payload = extract_json_payload('```JSON5\n{"a": 1}\n```')

    # then
    assert error_status is False
    assert payload == {"a": 1}


def test_extract_json_payload_when_fence_has_no_language_tag() -> None:
    # when
    error_status, payload = extract_json_payload("```\n[1, 2]\n```")

    # then
    assert error_status is False
    assert payload == [1, 2]


def test_extract_json_payload_when_only_closing_fence_present() -> None:
    # given - Qwen 3.8 Max shape
    raw = (
        "[\n"
        '\t{"bbox_2d": [46, 571, 997, 931], "label": "trailer"},\n'
        '\t{"bbox_2d": [1, 2, 3, 4], "label": "trailer"}\n'
        "]\n"
        "```"
    )

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == [
        {"bbox_2d": [46, 571, 997, 931], "label": "trailer"},
        {"bbox_2d": [1, 2, 3, 4], "label": "trailer"},
    ]


def test_extract_json_payload_when_only_opening_fence_present() -> None:
    # when
    error_status, payload = extract_json_payload('```json\n{"a": 1}')

    # then
    assert error_status is False
    assert payload == {"a": 1}


def test_extract_json_payload_when_json_lines_given() -> None:
    # given - GLM 5.3 Flash shape
    raw = (
        '{"bbox_2d": [618, 129, 644, 176], "label": "car"}\n'
        '{"bbox_2d": [656, 223, 679, 276], "label": "car"}\n'
        '{"bbox_2d": [641, 330, 682, 371], "label": "bus"}'
    )

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == [
        {"bbox_2d": [618, 129, 644, 176], "label": "car"},
        {"bbox_2d": [656, 223, 679, 276], "label": "car"},
        {"bbox_2d": [641, 330, 682, 371], "label": "bus"},
    ]


def test_extract_json_payload_when_comma_separated_objects_without_brackets_given() -> (
    None
):
    # given - Muse Glimmer shape
    raw = '{"a": 1}, {"a": 2},'

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == [{"a": 1}, {"a": 2}]


def test_extract_json_payload_when_json_lines_inside_stray_fences_given() -> None:
    # when
    error_status, payload = extract_json_payload('```json\n{"a": 1}\n{"a": 2}\n```')

    # then
    assert error_status is False
    assert payload == [{"a": 1}, {"a": 2}]


def test_extract_json_payload_when_list_wrapped_in_prose_given() -> None:
    # given - Z.ai shape
    raw = (
        "Here are the detected objects: "
        '[{"bbox_2d": [200, 100, 1000, 500], "label": "cat"}] '
        "Let me know if you need anything else."
    )

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == [{"bbox_2d": [200, 100, 1000, 500], "label": "cat"}]


def test_extract_json_payload_when_object_wrapped_in_prose_given() -> None:
    # when
    error_status, payload = extract_json_payload('Sure! {"class_name": "cat"} Done.')

    # then
    assert error_status is False
    assert payload == {"class_name": "cat"}


def test_extract_json_payload_when_object_sequence_has_garbage_between_entries() -> (
    None
):
    # given
    raw = '{"a": 1} some explanation {"a": 2}'

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is True
    assert payload == {}


@pytest.mark.parametrize(
    "raw",
    [
        "invalid",
        "",
        "42",
        '"just a string"',
        '```json\n[\n  {"box_',  # truncated at max_tokens
        '[{"a": 1}, {"a": 2}',  # unterminated array
    ],
)
def test_extract_json_payload_when_nothing_recoverable(raw: str) -> None:
    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is True
    assert payload == {}


def test_extract_json_payload_when_non_string_given() -> None:
    # when
    error_status, payload = extract_json_payload(None)

    # then
    assert error_status is True
    assert payload == {}


def test_extract_json_payload_when_sequence_of_arrays_given() -> None:
    # when
    error_status, payload = extract_json_payload('[{"a": 1}]\n[]\n[{"a": 2}]')

    # then
    assert error_status is False
    assert payload == [{"a": 1}, {"a": 2}]


def test_extract_json_payload_when_only_empty_arrays_given() -> None:
    # when
    error_status, payload = extract_json_payload("[]\n[]")

    # then
    assert error_status is False
    assert payload == []


def test_extract_json_payload_when_objects_end_with_stray_closing_bracket() -> None:
    # given - list body emitted without its opening bracket
    raw = '{"a": 1}, {"a": 2}]'

    # when
    error_status, payload = extract_json_payload(raw)

    # then
    assert error_status is False
    assert payload == [{"a": 1}, {"a": 2}]


def test_coerce_classification_payload_when_dict_given() -> None:
    # when
    result = coerce_classification_payload({"class_name": "cat", "confidence": 0.9})

    # then
    assert result == {"class_name": "cat", "confidence": 0.9}


def test_coerce_classification_payload_when_bare_class_list_given() -> None:
    # when
    result = coerce_classification_payload(
        [
            {"class": "cat", "confidence": 0.9},
            {"class_name": "dog"},
            {"label": "cow", "confidence": 0.2},
        ]
    )

    # then
    assert result == {
        "predicted_classes": [
            {"class": "cat", "confidence": 0.9},
            {"class": "dog", "confidence": 1.0},
            {"class": "cow", "confidence": 0.2},
        ]
    }


@pytest.mark.parametrize(
    "payload", [[], [{"confidence": 0.9}], ["cat"], [{"class": "cat"}, 3]]
)
def test_coerce_classification_payload_when_list_is_not_class_entries(
    payload: list,
) -> None:
    # when
    result = coerce_classification_payload(payload)

    # then
    assert result is None
