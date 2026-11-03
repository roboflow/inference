"""Tests for ``roboflow_core/databricks_sql_image_source@v1``.

The Databricks connector is replaced by an in-memory fake installed into
``sys.modules`` and DNS resolution is patched, so the tests need neither
network access nor the package.
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
from roboflow_workflows.core_steps.sinks.databricks.sql_image_source import v1
from roboflow_workflows.core_steps.sinks.databricks.sql_image_source.v1 import (
    BlockManifest,
    DatabricksSQLImageSourceBlockV1,
    decode_image,
    to_json_safe,
    validate_destination,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.utils.url_input import URLAddressNotAllowedError

HOST = "dbc-a1b2c3d4-e5f6.cloud.databricks.com"
HTTP_PATH = "/sql/1.0/warehouses/abcdef1234567890"


class FakeRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def asDict(self) -> Dict[str, Any]:
        return dict(self._data)


class FakeCursor:
    def __init__(self, rows: List[Dict[str, Any]], recorder: Dict[str, Any]):
        self._rows = rows
        self._recorder = recorder
        self._position = 0

    def execute(self, statement: str) -> None:
        self._recorder["statement"] = statement

    def fetchmany(self, size: int) -> List[FakeRow]:
        self._recorder.setdefault("fetchmany_sizes", []).append(size)
        batch = self._rows[self._position : self._position + size]
        self._position += size
        return [FakeRow(row) for row in batch]

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
def fake_databricks(
    rows: List[Dict[str, Any]], resolved_ips: List[str] = ("52.1.2.3",)
) -> Iterator[Dict[str, Any]]:
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
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=list(resolved_ips)
    ) as resolver:
        recorder["resolver"] = resolver
        yield recorder


def jpeg_bytes(height: int = 32, width: int = 48, value: int = 200) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    return buffer.tobytes()


def run_block(**overrides: Any) -> list:
    parameters = {
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
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
        "http_path": HTTP_PATH,
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


@pytest.mark.parametrize("max_rows", [0, 1_001, 10_000])
def test_manifest_parsing_when_max_rows_out_of_range(max_rows: int) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/databricks_sql_image_source@v1",
        "name": "db",
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
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
# Destination validation (SSRF guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host",
    [
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        "DBC-A1B2C3D4-E5F6.cloud.databricks.com",
        "adb-1234567890123456.7.azuredatabricks.net",
        "1234567890123456.7.gcp.databricks.com",
    ],
)
def test_validate_destination_accepts_databricks_hosts(host: str) -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips", return_value=["52.1.2.3"]):
        assert validate_destination(host, http_path=HTTP_PATH) == ["52.1.2.3"]


@pytest.mark.parametrize(
    "host",
    [
        "10.0.0.5",
        "169.254.169.254",
        "[::1]",
        "localhost",
        "internal-service.corp",
        "evil.com",
        "cloud.databricks.com.evil.com",
        "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com/path",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com:8443",
        "user@dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com\\evil",
        "",
    ],
)
def test_validate_destination_rejects_non_databricks_hosts(host: str) -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips") as resolver:
        with pytest.raises(ValueError):
            validate_destination(host, http_path=HTTP_PATH)
    resolver.assert_not_called()


@pytest.mark.parametrize(
    "http_path",
    [
        "/sql/1.0/warehouses/abcdef1234567890",
        "/sql/1.0/endpoints/abcdef1234567890",
        "/sql/protocolv1/o/1234567890123456/0123-456789-abcdefg1",
    ],
)
def test_validate_destination_accepts_warehouse_and_cluster_paths(
    http_path: str,
) -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips", return_value=["52.1.2.3"]):
        validate_destination(HOST, http_path=http_path)


@pytest.mark.parametrize(
    "http_path",
    [
        "/anything",
        "/sql/1.0/warehouses/../../admin",
        "sql/1.0/warehouses/abc",
        "",
        "/sql/1.0/warehouses/abc?x=1",
    ],
)
def test_validate_destination_rejects_other_paths(http_path: str) -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips") as resolver:
        with pytest.raises(ValueError, match="http_path"):
            validate_destination(HOST, http_path=http_path)
    resolver.assert_not_called()


def test_validate_destination_rejects_non_global_resolution() -> None:
    with mock.patch.object(
        v1,
        "resolve_and_validate_ips",
        side_effect=URLAddressNotAllowedError("resolves to non-global address"),
    ):
        with pytest.raises(ValueError, match="does not allow"):
            validate_destination(HOST, http_path=HTTP_PATH)


def test_validate_destination_rejects_unresolvable_host() -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips", side_effect=OSError("dns")):
        with pytest.raises(ValueError, match="Could not resolve"):
            validate_destination(HOST, http_path=HTTP_PATH)


def test_validate_destination_never_allows_non_global_on_hosted_runtime() -> None:
    with mock.patch.object(v1, "GCP_SERVERLESS", True), mock.patch.object(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", True
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=["52.1.2.3"]
    ) as resolver:
        validate_destination(HOST, http_path=HTTP_PATH)
    assert resolver.call_args.kwargs["allow_non_global_addresses"] is False


@pytest.mark.parametrize("operator_setting", [True, False])
def test_validate_destination_follows_operator_setting_when_self_hosted(
    operator_setting: bool,
) -> None:
    with mock.patch.object(v1, "GCP_SERVERLESS", False), mock.patch.object(
        v1, "LAMBDA", False
    ), mock.patch.object(
        v1, "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", operator_setting
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=["52.1.2.3"]
    ) as resolver:
        validate_destination(HOST, http_path=HTTP_PATH)
    assert resolver.call_args.kwargs["allow_non_global_addresses"] is operator_setting
    assert resolver.call_args.kwargs["port"] == 443


def test_run_validates_destination_before_connecting() -> None:
    with fake_databricks([{"id": 1, "bytes": jpeg_bytes()}]) as recorder:
        with pytest.raises(ValueError, match="not a Databricks workspace host"):
            run_block(server_hostname="internal.corp")
    assert "connect_kwargs" not in recorder


# ---------------------------------------------------------------------------
# Helpers and size limits
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


def test_decode_image_rejects_payload_over_byte_limit() -> None:
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", 100):
        with pytest.raises(ValueError, match="exceeds 100 bytes"):
            decode_image(jpeg_bytes())


def test_decode_image_rejects_base64_payload_over_byte_limit_before_decoding() -> None:
    encoded = base64.b64encode(jpeg_bytes()).decode()
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", 100), mock.patch.object(
        v1, "base64"
    ) as b64:
        with pytest.raises(ValueError, match="exceeds 100 bytes"):
            decode_image(encoded)
    b64.b64decode.assert_not_called()


def test_decode_image_rejects_image_over_pixel_limit_without_decoding() -> None:
    with mock.patch.object(v1, "MAX_IMAGE_PIXELS", 1_000), mock.patch.object(
        v1.cv2, "imdecode"
    ) as imdecode:
        with pytest.raises(ValueError, match="32x48|48x32"):
            decode_image(jpeg_bytes(height=32, width=48))
    imdecode.assert_not_called()


def test_to_json_safe_converts_non_json_types_and_drops_binary() -> None:
    assert to_json_safe(decimal.Decimal("1.5")) == 1.5
    assert to_json_safe(datetime.date(2026, 11, 3)) == "2026-11-03"
    assert to_json_safe({"a": [decimal.Decimal(2), b"x"], "b": b"y"}) == {"a": [2.0]}
    assert to_json_safe("plain") == "plain"


# ---------------------------------------------------------------------------
# Block execution
# ---------------------------------------------------------------------------


def test_run_emits_one_element_per_row_and_strips_binary_columns() -> None:
    # given
    rows = [
        {
            "id": 1,
            "bytes": jpeg_bytes(),
            "raccoon_name": "rocky",
            "weight": 4.2,
            "thumb": b"\x00",
        },
        {
            "id": 2,
            "bytes": jpeg_bytes(value=10),
            "raccoon_name": "bandit",
            "weight": 5.0,
            "thumb": b"\x00",
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
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
        "access_token": "dapi-secret",
        "_socket_timeout": v1.SOCKET_TIMEOUT_SECONDS,
    }
    assert recorder["statement"] == "SELECT * FROM t"
    assert recorder["connection_closed"] and recorder["cursor_closed"]


def test_run_fetches_in_batches_and_stops_at_max_rows() -> None:
    # given
    rows = [{"id": i, "bytes": jpeg_bytes()} for i in range(10)]

    # when
    with fake_databricks(rows) as recorder, mock.patch.object(
        v1, "FETCH_BATCH_SIZE", 4
    ):
        result = run_block(max_rows=7)

    # then
    assert [element["image_names"] for element in result] == [str(i) for i in range(7)]
    assert recorder["fetchmany_sizes"] == [4, 3]


def test_run_fails_when_decoded_pixel_budget_is_exhausted() -> None:
    # given
    rows = [{"id": i, "bytes": jpeg_bytes()} for i in range(3)]
    one_image = 32 * 48 * 3

    # when / then
    with fake_databricks(rows), mock.patch.object(
        v1, "MAX_TOTAL_DECODED_BYTES", one_image * 2
    ):
        with pytest.raises(ValueError, match="byte budget after 2 rows"):
            run_block()


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


def test_run_skips_undecodable_and_oversized_rows_by_default() -> None:
    # given
    rows = [
        {"id": 1, "bytes": b"junk"},
        {"id": 2, "bytes": jpeg_bytes()},
        {"id": 3, "bytes": jpeg_bytes(height=64, width=64)},
    ]

    # when
    with fake_databricks(rows), mock.patch.object(v1, "MAX_IMAGE_PIXELS", 2_000):
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
    with mock.patch.dict(
        sys.modules, {"databricks": None, "databricks.sql": None}
    ), mock.patch.object(v1, "resolve_and_validate_ips", return_value=["52.1.2.3"]):
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
            "server_hostname": HOST,
            "http_path": HTTP_PATH,
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
