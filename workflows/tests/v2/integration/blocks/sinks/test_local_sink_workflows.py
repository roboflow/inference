"""Compiled workflows with the local sinks and the data aggregator.

These run the real compiler, session and executor. Requests go only to an
in-process fake client; files go only to pytest's temporary directory::

    name, count ─▶ roboflow_core/csv_formatter@v1 ─▶ roboflow_core/local_file_sink@v1 ─▶ file
    count ─▶ roboflow_core/data_aggregator@v1 ─▶ count_sum, count_max
    count ─▶ roboflow_core/data_aggregator@v1 ─▶ roboflow_core/continue_if_boolean@v1(interval_complete)
          ─▶ roboflow_core/csv_formatter@v1 ─▶ roboflow_core/local_file_sink@v1 ─▶ file
    count ─▶ roboflow_core/webhook_sink@v1 ─▶ webhook_client (+ sink_effects)
"""

from pathlib import Path
from typing import Any, Dict, List

import pytest
from roboflow_workflows.v2.blocks.analytics.data_aggregator.v1 import DataAggregatorBlock
from roboflow_workflows.v2.blocks.control.continue_if_boolean.v1 import ContinueIfBlock
from roboflow_workflows.v2.blocks.formatters.csv_formatter.v1 import CsvFormatterBlock
from roboflow_workflows.v2.blocks.sinks.local_file_sink.v1 import LocalFileSinkBlock
from roboflow_workflows.v2.blocks.sinks.runtime import BackgroundEffects
from roboflow_workflows.v2.blocks.sinks.webhook_sink.v1 import WebhookSinkBlock
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import ResourceError, StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions

from tests.v2.support.sinks.loopback import RecordingWebhookClient

CATALOGUE = Catalogue(
    [
        CsvFormatterBlock,
        LocalFileSinkBlock,
        WebhookSinkBlock,
        DataAggregatorBlock,
        ContinueIfBlock,
    ]
)
NAME_INPUT = {"type": "WorkflowBatchInput", "name": "name", "kind": ["string"]}
COUNT_INPUT = {"type": "WorkflowBatchInput", "name": "count", "kind": ["integer"]}


def _output(name: str, selector: str) -> Dict[str, str]:
    return {"type": "JsonField", "name": name, "selector": selector}


def _compile(steps: List[Dict[str, Any]], outputs: List[Dict[str, str]], **options):
    definition = {
        "version": "2.0",
        "inputs": [
            NAME_INPUT,
            COUNT_INPUT,
            {"type": "WorkflowParameter", "name": "limit"},
        ],
        "steps": steps,
        "outputs": outputs,
    }
    plan = compile_workflow(
        definition, catalogue=CATALOGUE, options=CompileOptions(**options)
    )

    return plan


def _csv_to_file(directory: Path, **sink: Any) -> List[Dict[str, Any]]:
    steps = [
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv",
            "columns_data": {"name": "$inputs.name", "count": "$inputs.count"},
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "file",
            "content": "$steps.csv.csv_content",
            "output_mode": "append_log",
            "target_directory": str(directory),
            **sink,
        },
    ]

    return steps


def _webhook(**params: Any) -> Dict[str, Any]:
    step = {
        "type": "roboflow_core/webhook_sink@v1",
        "name": "hook",
        "url": "http://hooks.test/count",
        "method": "POST",
        "json_payload": {"name": "$inputs.name", "count": "$inputs.count"},
        "cooldown_seconds": 0,
        **params,
    }

    return step


def _inputs(limit: Any = 10) -> Dict[str, Any]:
    return {"name": ["a", "b", "c"], "count": [2, 5, 1], "limit": limit}


def _csv_bodies(directory: Path) -> List[List[str]]:
    bodies = [
        [line.split(",")[:2] for line in path.read_text().splitlines()]
        for path in sorted(directory.glob("*.csv"))
    ]

    return bodies


def test_append_log_is_written_although_its_outputs_are_not_requested(
    tmp_path: Path,
) -> None:
    # given: the sink's outputs exist, but the caller requests only the CSV
    plan = _compile(
        _csv_to_file(tmp_path),
        [
            _output("csv", "$steps.csv.csv_content"),
            _output("saved", "$steps.file.message"),
        ],
        requested_outputs=("csv",),
    )
    session = plan.create_session(resources={"allow_access_to_file_system": True})

    # when: two runs of three rows each
    for _ in range(2):
        session.run(_inputs())

    # then: one file, one header, six rows in order
    header, *rows = _csv_bodies(tmp_path)[0]
    assert header == ["name", "count"]
    assert rows == [["a", "2"], ["b", "5"], ["c", "1"]] * 2


def test_selected_max_entries_is_checked_at_run_time(tmp_path: Path) -> None:
    plan = _compile(
        _csv_to_file(tmp_path, max_entries_per_file="$inputs.limit"),
        [_output("saved", "$steps.file.error_status")],
    )
    session = plan.create_session(resources={"allow_access_to_file_system": True})

    with pytest.raises(
        StepExecutionError,
        match="max_entries_per_file is invalid.*greater than or equal to 1",
    ):
        session.run(_inputs(limit=0))

    assert list(tmp_path.iterdir()) == []


def test_file_writes_need_the_host_permission(tmp_path: Path) -> None:
    plan = _compile(_csv_to_file(tmp_path), [_output("saved", "$steps.file.message")])

    with pytest.raises(StepExecutionError, match="allow_access_to_file_system"):
        plan.create_session().run(_inputs())

    assert list(tmp_path.iterdir()) == []


def _aggregator(**params: Any) -> Dict[str, Any]:
    step = {
        "type": "roboflow_core/data_aggregator@v1",
        "name": "stats",
        "data": {"count": "$inputs.count"},
        "aggregation_mode": {"count": ["sum", "max"]},
        "interval_unit": "runs",
        "interval": 3,
        **params,
    }

    return step


def test_aggregator_outputs_statistics_on_the_interval_end() -> None:
    plan = _compile(
        [_aggregator()],
        [
            _output("done", "$steps.stats.interval_complete"),
            _output("total", "$steps.stats.count_sum"),
            _output("peak", "$steps.stats.count_max"),
        ],
    )

    rows = plan.create_session().run(_inputs()).rows()

    assert rows == [
        {"done": False, "total": None, "peak": None},
        {"done": False, "total": None, "peak": None},
        {"done": True, "total": 8, "peak": 5},
    ]


def test_gated_summary_log_gets_only_completed_intervals(tmp_path: Path) -> None:
    # given: the summary reaches the CSV only when the interval completes
    steps = [
        _aggregator(),
        {
            "type": "roboflow_core/continue_if_boolean@v1",
            "name": "gate",
            "condition": "$steps.stats.interval_complete",
            "next_steps": ["$steps.csv"],
        },
        {
            "type": "roboflow_core/csv_formatter@v1",
            "name": "csv",
            "columns_data": {
                "total": "$steps.stats.count_sum",
                "peak": "$steps.stats.count_max",
            },
        },
        {
            "type": "roboflow_core/local_file_sink@v1",
            "name": "file",
            "content": "$steps.csv.csv_content",
            "output_mode": "append_log",
            "target_directory": str(tmp_path),
        },
    ]
    plan = _compile(steps, [_output("saved", "$steps.file.message")])
    session = plan.create_session(resources={"allow_access_to_file_system": True})

    # when: two runs of three rows, so two intervals of three calls
    for _ in range(2):
        session.run(_inputs())

    # then: one header and one row per interval; no empty pre-interval rows
    assert _csv_bodies(tmp_path) == [[["total", "peak"], ["8", "5"], ["8", "5"]]]


def test_webhook_sends_one_request_per_row_without_demanded_outputs() -> None:
    client = RecordingWebhookClient()
    plan = _compile(
        [_webhook(fire_and_forget=False)],
        [_output("status", "$steps.hook.error_status")],
        requested_outputs=(),
    )

    plan.create_session(resources={"webhook_client": client}).run(_inputs())

    payloads = [request.json_payload for request in client.requests]
    assert payloads == [
        {"name": "a", "count": 2},
        {"name": "b", "count": 5},
        {"name": "c", "count": 1},
    ]


def test_fire_and_forget_requests_finish_when_the_host_drains() -> None:
    client = RecordingWebhookClient()
    plan = _compile([_webhook()], [_output("message", "$steps.hook.message")])

    with BackgroundEffects(max_pending=8) as effects:
        session = plan.create_session(
            resources={"webhook_client": client, "sink_effects": effects}
        )
        rows = session.run(_inputs()).rows()

    assert [row["message"] for row in rows] == [
        "Notification sent in the background task"
    ] * 3
    assert sorted(request.json_payload["count"] for request in client.requests) == [
        1,
        2,
        5,
    ]
    assert effects.stats().completed == 3


def test_webhook_without_a_host_client_fails_before_running() -> None:
    plan = _compile([_webhook()], [_output("message", "$steps.hook.message")])

    with pytest.raises(ResourceError, match="webhook_client"):
        plan.create_session()
