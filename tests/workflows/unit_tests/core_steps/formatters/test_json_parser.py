import json

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.formatters.json_parser.v1 import (
    BlockManifest,
    JSONParserBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import BOOLEAN_KIND


def test_parsing_manifest_when_input_is_valid() -> None:
    # given
    raw_manifest = {
        "name": "parser",
        "type": "roboflow_core/json_parser@v1",
        "raw_json": "$steps.some.a",
        "expected_fields": ["a", "b", "c"],
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        name="parser",
        type="roboflow_core/json_parser@v1",
        raw_json="$steps.some.a",
        expected_fields=["a", "b", "c"],
    )


def test_parsing_manifest_when_input_is_invalid() -> None:
    # given
    raw_manifest = {
        "name": "parser",
        "type": "roboflow_core/json_parser@v1",
        "raw_json": "$steps.some.a",
        "expected_fields": ["a", "b", "c", "error_status"],
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(raw_manifest)


def test_manifest_get_actual_outputs() -> None:
    # given
    manifest = BlockManifest(
        name="parser",
        type="roboflow_core/json_parser@v1",
        raw_json="$steps.some.a",
        expected_fields=["a", "b", "c"],
    )

    # when
    result = manifest.get_actual_outputs()

    # then
    assert result == [
        OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        OutputDefinition(name="a"),
        OutputDefinition(name="b"),
        OutputDefinition(name="c"),
    ]


def test_block_run_when_valid_json_given_and_all_fields_declared() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }


def test_block_run_when_valid_json_given_and_subset_of_fields_declared() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
    }


def test_block_run_when_valid_json_given_and_subset_of_declared_fields_found() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b", "c"])

    # then
    assert result == {
        "error_status": True,
        "a": "1",
        "b": "2",
        "c": None,
    }


def test_block_run_when_multiple_json_documents_provided() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json="\n".join([raw_json] * 2), expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": True,
        "a": None,
        "b": None,
    }


def test_block_run_when_invalid_json_provided() -> None:
    # given
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json="invalid", expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": True,
        "a": None,
        "b": None,
    }


def test_block_run_when_json_in_markdown_provided() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    raw_json = f"```json\n{raw_json}\n```"
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }


def test_block_run_when_indented_json_in_markdown_provided() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"}, indent=4)
    raw_json = f"```json\n{raw_json}\n```"
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }


def test_block_run_when_json_in_markdown_uppercase_provided() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    raw_json = f"```JSON\n{raw_json}\n```"
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }


def test_block_run_when_json_in_markdown_without_new_lines_provided() -> None:
    # given
    raw_json = json.dumps({"a": "1", "b": "2"})
    raw_json = f"```JSON{raw_json}```"
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }


def test_block_run_when_multiple_jsons_in_markdown_provided() -> None:
    # given
    raw_json_1 = json.dumps({"a": "1", "b": "2"})
    raw_json_2 = json.dumps({"a": "3", "b": "4"})
    raw_json = f"```json\n{raw_json_1}\n```\n``json\n{raw_json_2}\n```"
    block = JSONParserBlockV1()

    # when
    result = block.run(raw_json=raw_json, expected_fields=["a", "b"])

    # then
    assert result == {
        "error_status": False,
        "a": "1",
        "b": "2",
    }
