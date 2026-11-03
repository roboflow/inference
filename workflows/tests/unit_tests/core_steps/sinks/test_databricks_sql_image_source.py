"""Tests for ``roboflow_core/databricks_sql_image_source@v1``.

The Databricks connector is replaced by an in-memory fake installed into
``sys.modules``, so the tests need neither network access nor the package.
"""

import base64
import datetime
import decimal
import sys
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest import mock

import cv2
import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.sinks.databricks.sql_image_source.v1 import (
    BlockManifest,
    DatabricksSQLImageSourceBlockV1,
    decode_image,
    to_json_safe,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData


class FakeRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def asDict(self) -> Dict[str, Any]:
        return dict(self._data)


class FakeCursor:
    def __init__(self, rows: List[Dict[str, Any]], recorder: Dict[str, Any]):
        self._rows = rows
        self._recorder = recorder

    def execute(self, statement: str) -> None:
        self._recorder["statement"] = statement

    def fetchmany(self, size: int) -> List[FakeRow]:
        self._recorder["fetchmany_size"] = size
        return [FakeRow(row) for row in self._rows[:size]]

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        self._recorder["cursor_closed"] = True


class FakeConnection:
    def __init__(self, rows: List[Dict[str, Any]], recorder: Dict[str, Any]):
        self._rows = rows
        self._recorder = recorder

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._rows, self._recorder)

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        self._recorder["connection_closed"] = True


@contextmanager
def fake_databricks(rows: List[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    recorder: Dict[str, Any] = {}

    def connect(**kwargs: Any) -> FakeConnection:
        recorder["connect_kwargs"] = kwargs
        return FakeConnection(rows, recorder)

    sql_module = types.ModuleType("databricks.sql")
    sql_module.connect = connect
    databricks_module = types.ModuleType("databricks")
    databricks_module.sql = sql_module
    with mock.patch.dict(
        sys.modules, {"databricks": databricks_module, "databricks.sql": sql_module}
    ):
        yield recorder


def jpeg_bytes(height: int = 32, width: int = 48, value: int = 200) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


def run_block(**overrides: Any) -> list:
    parameters = {
        "server_hostname": "host",
        "http_path": "/sql/1.0/warehouses/x",
        "access_token": "dapi-secret",
        "sql_statement": "SELECT * FROM t",
        "image_column": "bytes",
        "image_name_column": "id",
        "max_rows": 100,
        "skip_undecodable": True,
    }
    parameters.update(overrides)
    return DatabricksSQLImageSourceBlockV1().run(**parameters)


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_parsing_when_input_is_valid() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/databricks_sql_image_source@v1",
        "name": "db",
        "server_hostname": "$inputs.databricks_host",
        "http_path": "/sql/1.0/warehouses/abc",
        "access_token": "$inputs.databricks_token",
        "sql_statement": "SELECT id, bytes FROM c.s.images",
        "image_name_column": "id",
        "max_rows": 5,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result.server_hostname == "$inputs.databricks_host"
    assert result.access_token == "$inputs.databricks_token"
    assert result.image_column == "bytes"
    assert result.max_rows == 5
    assert result.skip_undecodable is True


@pytest.mark.parametrize("max_rows", [0, 10_001])
def test_manifest_parsing_when_max_rows_out_of_range(max_rows: int) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/databricks_sql_image_source@v1",
        "name": "db",
        "server_hostname": "h",
        "http_path": "p",
        "access_token": "t",
        "sql_statement": "q",
        "max_rows": max_rows,
    }

    # when / then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(raw_manifest)


def test_manifest_declares_one_level_of_output_nesting() -> None:
    assert BlockManifest.get_output_dimensionality_offset() == 1
    assert {o.name for o in BlockManifest.describe_outputs()} == {
        "images",
        "image_names",
        "metadata",
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_decode_image_accepts_bytes_bytearray_memoryview_and_base64() -> None:
    raw = jpeg_bytes()
    payloads = [raw, bytearray(raw), memoryview(raw), base64.b64encode(raw).decode()]

    for payload in payloads:
        image = decode_image(payload)
        assert image.shape == (32, 48, 3)


@pytest.mark.parametrize("payload", [None, b"not an image", 42])
def test_decode_image_rejects_invalid_payloads(payload: Any) -> None:
    with pytest.raises(ValueError):
        decode_image(payload)


def test_to_json_safe_converts_non_json_types() -> None:
    assert to_json_safe(decimal.Decimal("1.5")) == 1.5
    assert to_json_safe(datetime.date(2026, 11, 3)) == "2026-11-03"
    assert to_json_safe(b"ab") == "YWI="
    assert to_json_safe({"a": [decimal.Decimal(2)]}) == {"a": [2.0]}
    assert to_json_safe("plain") == "plain"


# ---------------------------------------------------------------------------
# Block execution
# ---------------------------------------------------------------------------


def test_run_emits_one_element_per_row_and_strips_image_column() -> None:
    # given
    rows = [
        {"id": 1, "bytes": jpeg_bytes(), "raccoon_name": "rocky", "weight": 4.2},
        {
            "id": 2,
            "bytes": jpeg_bytes(value=10),
            "raccoon_name": "bandit",
            "weight": 5.0,
        },
    ]

    # when
    with fake_databricks(rows) as recorder:
        result = run_block()

    # then
    assert len(result) == 2
    assert [element["image_names"] for element in result] == ["1", "2"]
    assert result[0]["metadata"] == {"id": 1, "raccoon_name": "rocky", "weight": 4.2}
    assert isinstance(result[0]["images"], WorkflowImageData)
    assert result[0]["images"].numpy_image.shape == (32, 48, 3)
    assert recorder["connect_kwargs"] == {
        "server_hostname": "host",
        "http_path": "/sql/1.0/warehouses/x",
        "access_token": "dapi-secret",
    }
    assert recorder["statement"] == "SELECT * FROM t"
    assert recorder["fetchmany_size"] == 100
    assert recorder["connection_closed"] and recorder["cursor_closed"]


def test_run_generates_uuid_names_when_name_column_unset() -> None:
    # given
    rows = [{"bytes": jpeg_bytes()}, {"bytes": jpeg_bytes()}]

    # when
    with fake_databricks(rows):
        result = run_block(image_name_column=None)

    # then
    names = [element["image_names"] for element in result]
    assert len(set(names)) == 2
    assert all(len(name) == 36 for name in names)
    assert result[0]["metadata"] == {}


def test_run_skips_undecodable_rows_by_default() -> None:
    # given
    rows = [{"id": 1, "bytes": b"junk"}, {"id": 2, "bytes": jpeg_bytes()}]

    # when
    with fake_databricks(rows):
        result = run_block()

    # then
    assert [element["image_names"] for element in result] == ["2"]


def test_run_raises_on_undecodable_row_when_skipping_disabled() -> None:
    # given
    rows = [{"id": 1, "bytes": b"junk"}, {"id": 2, "bytes": jpeg_bytes()}]

    # when / then
    with fake_databricks(rows):
        with pytest.raises(ValueError, match="row 0"):
            run_block(skip_undecodable=False)


def test_run_fails_clearly_when_image_column_missing() -> None:
    # given
    rows = [{"id": 1, "content": jpeg_bytes()}]

    # when / then
    with fake_databricks(rows):
        with pytest.raises(ValueError, match="Column `bytes` not present"):
            run_block()


def test_run_gives_actionable_error_when_connector_missing() -> None:
    with mock.patch.dict(sys.modules, {"databricks": None, "databricks.sql": None}):
        with pytest.raises(RuntimeError, match="databricks-sql-connector"):
            run_block()


# ---------------------------------------------------------------------------
# Execution engine
# ---------------------------------------------------------------------------

SOURCE_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "databricks_token"},
        {"type": "WorkflowParameter", "name": "sql"},
    ],
    "steps": [
        {
            "type": "roboflow_core/databricks_sql_image_source@v1",
            "name": "db",
            "server_hostname": "dbc-test.cloud.databricks.com",
            "http_path": "/sql/1.0/warehouses/abc",
            "access_token": "$inputs.databricks_token",
            "sql_statement": "$inputs.sql",
            "image_name_column": "id",
            "max_rows": 5,
        }
    ],
    "outputs": [
        {"type": "JsonField", "name": "images", "selector": "$steps.db.images"},
        {"type": "JsonField", "name": "names", "selector": "$steps.db.image_names"},
        {"type": "JsonField", "name": "metadata", "selector": "$steps.db.metadata"},
    ],
}


def test_engine_runs_block_as_input_free_step_producing_a_batch() -> None:
    # given
    rows = [
        {"id": 7, "bytes": jpeg_bytes(), "raccoon_name": "rocky"},
        {"id": 8, "bytes": jpeg_bytes(), "raccoon_name": "bandit"},
        {"id": 9, "bytes": jpeg_bytes(), "raccoon_name": "meeko"},
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=SOURCE_WORKFLOW,
        init_parameters={
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )

    # when
    with fake_databricks(rows) as recorder:
        result = execution_engine.run(
            runtime_parameters={"databricks_token": "dapi-xyz", "sql": "SELECT 1"}
        )

    # then
    assert recorder["connect_kwargs"]["access_token"] == "dapi-xyz"
    assert recorder["statement"] == "SELECT 1"
    assert recorder["fetchmany_size"] == 5
    assert len(result) == 1, "Input-free step runs exactly once"
    assert result[0]["names"] == ["7", "8", "9"]
    assert [m["raccoon_name"] for m in result[0]["metadata"]] == [
        "rocky",
        "bandit",
        "meeko",
    ]
    assert len(result[0]["images"]) == 3
    assert all(isinstance(image, WorkflowImageData) for image in result[0]["images"])


def test_engine_returns_empty_lists_when_query_yields_no_rows() -> None:
    # given
    execution_engine = ExecutionEngine.init(
        workflow_definition=SOURCE_WORKFLOW,
        init_parameters={
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=1,
    )

    # when
    with fake_databricks([]):
        result = execution_engine.run(
            runtime_parameters={"databricks_token": "dapi-xyz", "sql": "SELECT 1"}
        )

    # then
    assert len(result) == 1
    assert result[0]["images"] == []
    assert result[0]["names"] == []
    assert result[0]["metadata"] == []
