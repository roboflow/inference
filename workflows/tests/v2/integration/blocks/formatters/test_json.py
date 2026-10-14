"""Compiled workflows with ``roboflow_core/json_parser@v1`` and real product consumers.

These run the real compiler, session and executor. Parsed fields declare no
kind, so a typed consumer accepts the selector at compile time and checks the
value when the step runs::

    text ─▶ roboflow_core/json_parser@v1 ─┬─ width, height ─▶ roboflow_core/resize@v1 (INTEGER params)
                               └─ error_status (BOOLEAN)
"""

from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.image.primitives.resize.v1 import ResizeBlock
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    KindMismatchError,
    ParamsValidationError,
    SelectorError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

MODES = ("run", "phases")
TEXT_INPUT = {"type": "WorkflowBatchInput", "name": "text", "kind": ["string"]}
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
LLM_ANSWER = 'Here you go:\n```json\n{"output width": 40, "height": 20}\n```'
SIZE_FIELDS = {"width": "output width", "height": "height"}


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _parser(name: str, raw_json: str, expected_fields: Any) -> Dict[str, Any]:
    step = {
        "type": "roboflow_core/json_parser@v1",
        "name": name,
        "raw_json": raw_json,
        "expected_fields": expected_fields,
    }

    return step


def _resize(width: str, height: str) -> Dict[str, Any]:
    step = {
        "type": "roboflow_core/resize@v1",
        "name": "resize",
        "image": "$inputs.image",
        "width": width,
        "height": height,
        "interpolation": "nearest",
    }

    return step


def _compile(definition: Dict[str, Any], *, mode: str = "run"):
    catalogue = Catalogue([JsonParserBlock, ResizeBlock])
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _images(count: int) -> List[torch.Tensor]:
    return [torch.zeros((3, 60, 80), dtype=torch.uint8) for _ in range(count)]


def _resize_definition() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT, IMAGE_INPUT],
        "steps": [
            _parser("size", "$inputs.text", SIZE_FIELDS),
            _resize("$steps.size.width", "$steps.size.height"),
        ],
        "outputs": [
            _output("resized", "$steps.resize.image"),
            _output("failed", "$steps.size.error_status"),
        ],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_parsed_fields_drive_a_typed_consumer_per_row(mode: str) -> None:
    # given: a fenced answer with a key containing a space, a raw object, and
    # text that is not JSON
    texts = [LLM_ANSWER, '{"output width": 16, "height": 8}', "no idea"]
    plan = _compile(_resize_definition(), mode=mode)

    # when
    rows = plan.create_session().run({"text": texts, "image": _images(3)}).rows()

    # then: the failed row resizes nothing, because its width is None
    assert [row["failed"] for row in rows] == [False, False, True]
    assert rows[0]["resized"].size_hw == (20, 40)
    assert rows[1]["resized"].size_hw == (8, 16)
    assert rows[2]["resized"] is None


def test_wrong_value_type_fails_in_the_typed_consumer_at_run_time() -> None:
    # given: compiles, because the parsed field declares no kind
    plan = _compile(_resize_definition())

    # when / then
    with pytest.raises(StepExecutionError, match="not a valid 'integer'"):
        plan.create_session().run(
            {"text": ['{"output width": "wide", "height": 8}'], "image": _images(1)}
        )


def test_wildcard_output_holds_every_mapped_name() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [
            _parser(
                "p", "$inputs.text", {"label": "class name", "upstream": "error_status"}
            )
        ],
        "outputs": [_output("all", "$steps.p.*")],
    }
    text = '{"class name": "car", "error_status": "timeout"}'

    # when
    rows = _compile(definition).create_session().run({"text": [text]}).rows()

    # then
    assert rows == [
        {"all": {"label": "car", "upstream": "timeout", "error_status": False}}
    ]


def test_missing_input_row_skips_the_parser() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [_parser("p", "$inputs.text", ["a"])],
        "outputs": [_output("all", "$steps.p.*")],
    }

    # when
    rows = _compile(definition).create_session().run({"text": ['{"a": 1}', None]})

    # then
    assert rows.rows() == [{"all": {"a": 1, "error_status": False}}, {"all": None}]


def test_parsed_string_feeds_a_second_parser() -> None:
    # given: the outer JSON carries the inner JSON as a string
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [
            _parser("outer", "$inputs.text", ["payload"]),
            _parser("inner", "$steps.outer.payload", ["score"]),
        ],
        "outputs": [
            _output("score", "$steps.inner.score"),
            _output("failed", "$steps.inner.error_status"),
        ],
    }
    text = '{"payload": "{\\"score\\": 0.75}"}'

    # when
    rows = _compile(definition).create_session().run({"text": [text]}).rows()

    # then
    assert rows == [{"score": 0.75, "failed": False}]


def test_boolean_error_status_cannot_feed_a_string_parameter() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [
            _parser("first", "$inputs.text", ["a"]),
            _parser("second", "$steps.first.error_status", ["b"]),
        ],
        "outputs": [_output("b", "$steps.second.b")],
    }

    # when / then
    with pytest.raises(KindMismatchError):
        _compile(definition)


def test_selector_to_an_unexpected_field_fails_compilation() -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [_parser("p", "$inputs.text", ["model_id"])],
        "outputs": [_output("confidence", "$steps.p.confidence")],
    }

    # when / then
    with pytest.raises(SelectorError, match="no output 'confidence'"):
        _compile(definition)


@pytest.mark.parametrize(
    "expected_fields, message",
    [
        (["output width"], "{'output_width': 'output width'}"),
        (["error_status"], "'error_status' is reserved"),
        (["a", "a"], "duplicate expected_fields entries"),
        ("$inputs.fields", "valid list or Input should be a valid dictionary"),
    ],
)
def test_invalid_expected_fields_fail_compilation_at_the_step(
    expected_fields: Any, message: str
) -> None:
    # given
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [_parser("p", "$inputs.text", expected_fields)],
        "outputs": [_output("status", "$steps.p.error_status")],
    }

    # when / then
    with pytest.raises(ParamsValidationError, match=r"\$steps\.p ") as error:
        _compile(definition)
    assert message in str(error.value)


def test_nested_workflow_exposes_parsed_fields_to_the_parent() -> None:
    # given: the child parses; the parent resizes with the child's outputs
    child = {
        "version": "2.0",
        "inputs": [TEXT_INPUT],
        "steps": [_parser("size", "$inputs.text", SIZE_FIELDS)],
        "outputs": [
            _output("width", "$steps.size.width"),
            _output("height", "$steps.size.height"),
            _output("failed", "$steps.size.error_status"),
        ],
    }
    definition = {
        "version": "2.0",
        "inputs": [TEXT_INPUT, IMAGE_INPUT],
        "steps": [
            {
                "type": "inner_workflow",
                "name": "child",
                "workflow_definition": child,
                "parameter_bindings": {"text": "$inputs.text"},
            },
            _resize("$steps.child.width", "$steps.child.height"),
        ],
        "outputs": [
            _output("resized", "$steps.resize.image"),
            _output("failed", "$steps.child.failed"),
        ],
    }
    texts = [LLM_ANSWER, '{"height": 8}']

    # when
    rows = (
        _compile(definition)
        .create_session()
        .run({"text": texts, "image": _images(2)})
        .rows()
    )

    # then
    assert rows[0]["resized"].size_hw == (20, 40)
    assert rows[0]["failed"] is False
    assert rows[1] == {"resized": None, "failed": True}
