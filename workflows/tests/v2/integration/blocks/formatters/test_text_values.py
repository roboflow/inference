"""Compiled workflows with the text and value formatters.

Fake detector (``tests/v2/support/models``); not real inference. Its row count
follows mean brightness: 0 -> 0 rows, 64 -> 1, 128 -> 2, 255 -> 4; classes
cycle person, car, dog. A fixed wall clock makes time outputs exact::

    image ─▶ roboflow_core/roboflow_object_detection_model@v3 ─▶ roboflow_core/property_definition@v1 (classes)
                                          ├─▶ roboflow_core/string_template@v1   per image
                                          └─▶ roboflow_core/csv_formatter@v1     one CSV
"""

import csv
import io
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.v2.blocks.control.continue_if.v1 import ContinueIfQueryBlock
from roboflow_workflows.v2.blocks.formatters import BLOCKS as FORMATTER_BLOCKS
from roboflow_workflows.v2.blocks.formatters.csv_formatter.v1 import CsvFormatterBlock
from roboflow_workflows.v2.blocks.formatters.json_parser.v1 import JsonParserBlock
from roboflow_workflows.v2.blocks.formatters.string_template.v1 import StringTemplateBlock
from roboflow_workflows.v2.blocks.formatters.current_time.v1 import CurrentTimeBlock
from roboflow_workflows.v2.blocks.formatters.first_non_empty_or_default.v1 import FirstNonEmptyOrDefaultBlock
from roboflow_workflows.v2.blocks.image import DynamicCropBlock
from roboflow_workflows.v2.blocks.models import BLOCKS as MODEL_BLOCKS
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.models.fake_task_models import NmsDetector

MODES = ("run", "phases")
START = datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc)
IMAGE_INPUT = {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}
NAME_INPUT = {"type": "WorkflowBatchInput", "name": "name", "kind": ["string"]}
SITE_INPUT = {"type": "WorkflowParameter", "name": "site", "default_value": "gate"}
CLASS_NAMES = [{"type": "DetectionsPropertyExtract", "property_name": "class_name"}]
JOIN = [{"type": "SequenceJoin", "separator": " "}]
LENGTH = [{"type": "SequenceLength"}]


class StepClock:
    """START, START + 1 s, ... in the requested timezone."""

    def __init__(self) -> None:
        self.reads = 0

    def __call__(self, tz):
        moment = START + timedelta(seconds=self.reads)
        self.reads += 1

        return moment.astimezone(tz)


def _compile(definition: Dict[str, Any], *, mode: str = "run"):
    catalogue = Catalogue(
        [
            *MODEL_BLOCKS,
            *FORMATTER_BLOCKS,
            StringTemplateBlock,
            CsvFormatterBlock,
            JsonParserBlock,
            FirstNonEmptyOrDefaultBlock,
            CurrentTimeBlock,
            ContinueIfQueryBlock,
            DynamicCropBlock,
        ],
        kinds=NATIVE_KINDS,
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


def _session(definition: Dict[str, Any], *, mode: str = "run", clock=None):
    resources = {"models": {"od": NmsDetector()}, "wall_clock": clock or StepClock()}
    session = _compile(definition, mode=mode).create_session(resources)

    return session


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _image(brightness: int, *, name: str) -> ImageData:
    pixels = torch.full((3, 20, 30), brightness, dtype=torch.uint8)

    return ImageData.from_tensor(pixels, image_id=name)


def _parse(document: str) -> List[List[str]]:
    return list(csv.reader(io.StringIO(document)))


DETECT_STEPS = [
    {
        "type": "roboflow_core/roboflow_object_detection_model@v3",
        "name": "detector",
        "image": "$inputs.image",
        "model_id": "od",
        "confidence": 0.0,
    },
    {
        "type": "roboflow_core/property_definition@v1",
        "name": "classes",
        "data": "$steps.detector.predictions",
        "operations": CLASS_NAMES,
    },
]


def _report_workflow() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, NAME_INPUT, SITE_INPUT],
        "steps": [
            *DETECT_STEPS,
            {
                "type": "roboflow_core/string_template@v1",
                "name": "summary",
                "template": "{image} at {site}: {count} objects [{classes}]",
                "data": {
                    "image": "$inputs.name",
                    "site": "$inputs.site",
                    "count": "$steps.classes.output",
                    "classes": "$steps.classes.output",
                },
                "data_operations": {"count": LENGTH, "classes": JOIN},
            },
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "report",
                "columns_data": {
                    "image": "$inputs.name",
                    "site": "$inputs.site",
                    "classes": "$steps.classes.output",
                    "count": "$steps.classes.output",
                },
                "columns_operations": {"count": LENGTH},
            },
        ],
        "outputs": [
            _output("summary", "$steps.summary.output"),
            _output("report", "$steps.report.csv_content"),
        ],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_a_detection_batch_becomes_messages_and_one_csv(mode: str) -> None:
    session = _session(_report_workflow(), mode=mode)
    images = [_image(128, name="a"), _image(0, name="b"), _image(255, name="c")]

    rows = session.run({"image": images, "name": ["a.jpg", "b.jpg", "c.jpg"]}).rows()

    assert [row["summary"] for row in rows] == [
        "a.jpg at gate: 2 objects [person car]",
        "b.jpg at gate: 0 objects []",
        "c.jpg at gate: 4 objects [person car dog person]",
    ]
    assert [row["report"] for row in rows[:2]] == [None, None]
    assert _parse(rows[2]["report"]) == [
        ["image", "site", "classes", "count", "timestamp"],
        ["a.jpg", "gate", "['person', 'car']", "2", "2026-07-01T10:00:00+00:00"],
        ["b.jpg", "gate", "[]", "0", "2026-07-01T10:00:01+00:00"],
        [
            "c.jpg",
            "gate",
            "['person', 'car', 'dog', 'person']",
            "4",
            "2026-07-01T10:00:02+00:00",
        ],
    ]


@pytest.mark.parametrize("mode", MODES)
def test_a_reused_session_starts_a_new_csv_document_for_each_run(mode: str) -> None:
    clock = StepClock()
    session = _session(_report_workflow(), mode=mode, clock=clock)

    first = session.run({"image": [_image(0, name="a")], "name": ["a"]}).rows()
    second = session.run({"image": [_image(0, name="b")], "name": ["b"]}).rows()

    header = ["image", "site", "classes", "count", "timestamp"]
    assert _parse(first[0]["report"]) == [
        header,
        ["a", "gate", "[]", "0", "2026-07-01T10:00:00+00:00"],
    ]
    assert _parse(second[0]["report"]) == [
        header,
        ["b", "gate", "[]", "0", "2026-07-01T10:00:01+00:00"],
    ]
    assert clock.reads == 2


def test_a_parameter_only_csv_is_one_document_per_run() -> None:
    definition = {
        "version": "2.0",
        "inputs": [SITE_INPUT],
        "steps": [
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "report",
                "columns_data": {"site": "$inputs.site", "note": '=1+1, "x"'},
            }
        ],
        "outputs": [_output("report", "$steps.report.csv_content")],
    }

    rows = _session(definition).run({"site": "dock"}).rows()

    assert rows == [
        {
            "report": (
                'site,note,timestamp\ndock,"\'=1+1, ""x""",'
                "2026-07-01T10:00:00+00:00\n"
            )
        }
    ]


def _gated_workflow() -> Dict[str, Any]:
    at_least_two = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {
                    "type": "DynamicOperand",
                    "operand_name": "classes",
                    "operations": LENGTH,
                },
                "comparator": {"type": "(Number) >="},
                "right_operand": {"type": "StaticOperand", "value": 2},
            }
        ],
    }
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, NAME_INPUT],
        "steps": [
            *DETECT_STEPS,
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "busy",
                "condition_statement": at_least_two,
                "evaluation_parameters": {"classes": "$steps.classes.output"},
                "next_steps": ["$steps.alert", "$steps.busy_log"],
            },
            {
                "type": "roboflow_core/string_template@v1",
                "name": "alert",
                "template": "ALERT {image}",
                "data": {"image": "$inputs.name"},
            },
            {
                "type": "roboflow_core/first_non_empty_or_default@v1",
                "name": "status",
                "data": ["$steps.alert.output"],
                "default": "quiet",
            },
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "busy_log",
                "columns_data": {"image": "$inputs.name"},
            },
        ],
        "outputs": [
            _output("status", "$steps.status.output"),
            _output("busy_log", "$steps.busy_log.csv_content"),
        ],
    }

    return definition


@pytest.mark.parametrize("mode", MODES)
def test_a_gated_branch_merges_with_a_default_and_skips_csv_rows(mode: str) -> None:
    session = _session(_gated_workflow(), mode=mode)
    images = [
        _image(255, name="a"),
        _image(64, name="b"),
        _image(128, name="c"),
        _image(0, name="d"),
    ]

    rows = session.run({"image": images, "name": ["a", "b", "c", "d"]}).rows()

    assert [row["status"] for row in rows] == ["ALERT a", "quiet", "ALERT c", "quiet"]
    # The document lands on the last invocation that ran, image c.
    assert [row["busy_log"] is None for row in rows] == [True, True, False, True]
    assert [line[0] for line in _parse(rows[2]["busy_log"])] == ["image", "a", "c"]


@pytest.mark.parametrize("mode", MODES)
def test_an_all_gated_batch_produces_no_csv_document_or_clock_reads(mode: str) -> None:
    clock = StepClock()
    session = _session(_gated_workflow(), mode=mode, clock=clock)

    rows = session.run(
        {"image": [_image(0, name="a"), _image(64, name="b")], "name": ["a", "b"]}
    ).rows()

    assert rows == [
        {"status": "quiet", "busy_log": None},
        {"status": "quiet", "busy_log": None},
    ]
    assert clock.reads == 0


@pytest.mark.parametrize("mode", MODES)
def test_template_renders_present_none_but_skips_an_unavailable_row(mode: str) -> None:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "text", "kind": ["string"]}],
        "steps": [
            {
                "type": "roboflow_core/json_parser@v1",
                "name": "record",
                "raw_json": "$inputs.text",
                "expected_fields": ["a", "b"],
            },
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "valid",
                "condition_statement": {
                    "type": "StatementGroup",
                    "statements": [
                        {
                            "type": "BinaryStatement",
                            "left_operand": {
                                "type": "DynamicOperand",
                                "operand_name": "failed",
                            },
                            "comparator": {"type": "=="},
                            "right_operand": {"type": "StaticOperand", "value": False},
                        }
                    ],
                },
                "evaluation_parameters": {"failed": "$steps.record.error_status"},
                "next_steps": ["$steps.selected"],
            },
            {
                "type": "roboflow_core/string_template@v1",
                "name": "selected",
                "template": "{a}",
                "data": {"a": "$steps.record.a"},
            },
            {
                "type": "roboflow_core/string_template@v1",
                "name": "message",
                "template": "{a}|{b}",
                "data": {"a": "$steps.selected.output", "b": "$steps.record.b"},
            },
        ],
        "outputs": [_output("message", "$steps.message.output")],
    }

    rows = (
        _session(definition, mode=mode)
        .run({"text": ['{"a": 1, "b": null}', "not JSON", '{"a": 2, "b": 3}']})
        .rows()
    )

    # V1's str.format renders a present None as text. The invalid JSON row has
    # no selected output: admission must skip it, not render "None|None".
    assert rows == [{"message": "1|None"}, {"message": None}, {"message": "2|3"}]


@pytest.mark.parametrize("mode", MODES)
def test_none_cells_keep_their_rows_while_gated_records_add_none(mode: str) -> None:
    keep_is_true = {
        "type": "StatementGroup",
        "statements": [
            {
                "type": "BinaryStatement",
                "left_operand": {"type": "DynamicOperand", "operand_name": "keep"},
                "comparator": {"type": "=="},
                "right_operand": {"type": "StaticOperand", "value": True},
            }
        ],
    }
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "text", "kind": ["string"]}],
        "steps": [
            {
                "type": "roboflow_core/json_parser@v1",
                "name": "record",
                "raw_json": "$inputs.text",
                "expected_fields": ["name", "v", "n", "keep"],
            },
            {
                "type": "roboflow_core/continue_if@v1",
                "name": "gate",
                "condition_statement": keep_is_true,
                "evaluation_parameters": {"keep": "$steps.record.keep"},
                "next_steps": ["$steps.log"],
            },
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "log",
                "columns_data": {
                    "name": "$steps.record.name",
                    "v": "$steps.record.v",
                    "n": "$steps.record.n",
                    "site": "dock",
                },
            },
        ],
        "outputs": [_output("log", "$steps.log.csv_content")],
    }
    records = [
        '{"name": "null", "v": null, "n": 1, "keep": true}',
        '{"name": "gated", "v": 7, "n": 2, "keep": false}',
        '{"name": "missing", "n": 3, "keep": true}',
        '{"name": "empty", "v": "", "n": null, "keep": true}',
        '{"name": "gated last", "v": 5, "n": 5, "keep": false}',
    ]

    rows = _session(definition, mode=mode).run({"text": records}).rows()

    # The document lands on the last record that ran; gated ones read no clock.
    assert [row["log"] is None for row in rows] == [True, True, True, False, True]
    assert _parse(rows[3]["log"]) == [
        ["name", "v", "n", "site", "timestamp"],
        ["null", "", "1", "dock", "2026-07-01T10:00:00+00:00"],
        ["missing", "", "3", "dock", "2026-07-01T10:00:01+00:00"],
        ["empty", "", "", "dock", "2026-07-01T10:00:02+00:00"],
    ]


def test_crops_of_ragged_detections_give_one_row_per_crop() -> None:
    definition = {
        "version": "2.0",
        "inputs": [IMAGE_INPUT, NAME_INPUT],
        "steps": [
            DETECT_STEPS[0],
            {
                "type": "roboflow_core/dynamic_crop@v1",
                "name": "crop",
                "image": "$inputs.image",
                "predictions": "$steps.detector.predictions",
            },
            {
                "type": "roboflow_core/property_definition@v1",
                "name": "crop_class",
                "data": "$steps.crop.predictions",
                "operations": CLASS_NAMES,
            },
            {
                "type": "roboflow_core/string_template@v1",
                "name": "label",
                "template": "{image}/{label}",
                "data": {"image": "$inputs.name", "label": "$steps.crop_class.output"},
                "data_operations": {"label": JOIN},
            },
            {
                "type": "roboflow_core/csv_formatter@v1",
                "name": "crops",
                "columns_data": {
                    "image": "$inputs.name",
                    "label": "$steps.label.output",
                },
            },
        ],
        "outputs": [
            _output("labels", "$steps.label.output"),
            _output("crops", "$steps.crops.csv_content"),
        ],
    }
    images = [_image(128, name="a"), _image(0, name="b"), _image(64, name="c")]

    rows = _session(definition).run({"image": images, "name": ["a", "b", "c"]}).rows()

    assert [row["labels"] for row in rows] == [["a/person", "a/car"], [], ["c/person"]]
    documents = [row["crops"] for row in rows]
    assert documents[0][:-1] == [None]
    assert _parse(documents[2][-1]) == [
        ["image", "label", "timestamp"],
        ["a", "a/person", "2026-07-01T10:00:00+00:00"],
        ["a", "a/car", "2026-07-01T10:00:01+00:00"],
        ["c", "c/person", "2026-07-01T10:00:02+00:00"],
    ]


def _clock_workflow() -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "tz", "default_value": "UTC"}],
        "steps": [
            {"type": "roboflow_core/current_time@v1", "name": "now", "timezone": "$inputs.tz"},
            {
                "type": "roboflow_core/string_template@v1",
                "name": "stamp",
                "template": "{date} {time} ({iso})",
                "data": {
                    "date": "$steps.now.date",
                    "time": "$steps.now.time",
                    "iso": "$steps.now.iso_string",
                },
            },
        ],
        "outputs": [
            _output("stamp", "$steps.stamp.output"),
            _output("timestamp", "$steps.now.timestamp"),
        ],
    }

    return definition


def test_current_time_feeds_a_template_in_the_selected_timezone() -> None:
    session = _session(_clock_workflow())

    rows = session.run({"tz": "America/New_York"}).rows()

    assert rows[0]["stamp"] == "2026-07-01 06:00:00 (2026-07-01T06:00:00-04:00)"
    assert rows[0]["timestamp"] == START


def test_a_selected_timezone_outside_the_curated_list_fails_the_run() -> None:
    session = _session(_clock_workflow())

    with pytest.raises(StepExecutionError, match="unsupported timezone 'Europe/Paris'"):
        session.run({"tz": "Europe/Paris"})


def test_a_selected_template_is_checked_against_data_when_it_arrives() -> None:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "template"},
            SITE_INPUT,
        ],
        "steps": [
            {
                "type": "roboflow_core/string_template@v1",
                "name": "text",
                "template": "$inputs.template",
                "data": {"site": "$inputs.site"},
            }
        ],
        "outputs": [_output("text", "$steps.text.output")],
    }
    session = _session(definition)

    assert session.run({"template": "Site {site}"}).rows() == [{"text": "Site gate"}]
    with pytest.raises(StepExecutionError, match=r"uses \['camera'\]"):
        session.run({"template": "{camera}"})


@pytest.mark.parametrize(
    ("step", "message"),
    [
        (
            {"type": "roboflow_core/string_template@v1", "template": "{missing}"},
            r"uses \['missing'\]",
        ),
        (
            {"type": "roboflow_core/csv_formatter@v1", "columns_data": {"timestamp": 1}},
            "reserved column name",
        ),
        (
            {
                "type": "roboflow_core/csv_formatter@v1",
                "columns_data": {"a": 1},
                "columns_operations": {"b": LENGTH},
            },
            "columns_operations names `b`",
        ),
        ({"type": "roboflow_core/current_time@v1", "timezone": "Mars/Base"}, "timezone"),
        ({"type": "roboflow_core/first_non_empty_or_default@v1", "data": []}, "at least 1"),
    ],
)
def test_invalid_definitions_fail_workflow_compilation(
    step: dict, message: str
) -> None:
    definition = {
        "version": "2.0",
        "inputs": [],
        "steps": [{**step, "name": "step"}],
        "outputs": [],
    }

    with pytest.raises(ParamsValidationError, match=message):
        _compile(definition)
