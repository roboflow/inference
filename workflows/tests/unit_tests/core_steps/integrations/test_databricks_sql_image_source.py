"""Tests for ``roboflow_core/databricks_sql_image_source@v1``.

The Databricks connector is replaced by an in-memory fake installed into
``sys.modules`` and DNS resolution is patched, so the tests need neither
network access nor the package. ``test_databricks_connector_contract.py``
covers the real connector.
"""

import base64
import datetime
import decimal
import io
import json
import sys
import threading
import time
import types
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest import mock

import cv2
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import v1
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source.v1 import (
    BlockManifest,
    DatabricksSQLImageSourceBlockV1,
    RunBudget,
    build_metadata,
    classify_http_path,
    connection_kwargs,
    decode_image,
    normalise_hostname,
    to_json_safe,
    validate_destination,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine
from roboflow_workflows.execution_engine.entities.base import WorkflowImageData
from roboflow_workflows.utils.url_input import URLAddressNotAllowedError

HOST = "dbc-a1b2c3d4-e5f6.cloud.databricks.com"
HTTP_PATH = "/sql/1.0/warehouses/abcdef1234567890"
PUBLIC_IP = "52.1.2.3"


class FakeRow:
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def asDict(self) -> Dict[str, Any]:
        return dict(self._data)


class FakeCursor:
    """Record connector calls and expose deterministic fixture rows."""

    def __init__(self, rows: List[Dict[str, Any]], recorder: Dict[str, Any]):
        self._rows = rows
        self._recorder = recorder
        self._position = 0

    def execute(self, statement: str, parameters=None) -> None:
        """Record a query and its separate bound parameters.

        Args:
            statement (str): SQL template submitted by the block.
            parameters (Optional[dict]): Native bound parameters supplied to the connector.
        """
        self._recorder["statement"] = statement
        self._recorder["parameters"] = parameters
        delay = self._recorder.get("execute_delay")
        if delay:
            time.sleep(delay)

    def fetchmany(self, size: int) -> List[FakeRow]:
        self._recorder.setdefault("fetchmany_sizes", []).append(size)
        delay = self._recorder.get("fetch_delay")
        if delay:
            time.sleep(delay)
        batch = self._rows[self._position : self._position + size]
        self._position += size
        return [FakeRow(row) for row in batch]

    def cancel(self) -> None:
        self._recorder["cancelled"] = True

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *args: Any) -> None:
        self._recorder["cursor_closed"] = True


class FakeConnection:
    def __init__(self, rows: List[Dict[str, Any]], recorder: Dict[str, Any]):
        self._rows = rows
        self._recorder = recorder

    def cursor(self, **kwargs: Any) -> FakeCursor:
        self._recorder["cursor_kwargs"] = kwargs
        return FakeCursor(self._rows, self._recorder)

    def close(self) -> None:
        self._recorder["connection_closed"] = True

    def __enter__(self) -> "FakeConnection":
        return self

    def __exit__(self, *args: Any) -> None:
        self._recorder["connection_closed"] = True


@contextmanager
def fake_databricks(
    rows: List[Dict[str, Any]],
    *,
    resolved_ips: List[str] = (PUBLIC_IP,),
    recorder: Dict[str, Any] = None,
) -> Iterator[Dict[str, Any]]:
    """Install a fake connector and resolver while recording block side effects.

    Args:
        rows (List[Dict[str, Any]]): Warehouse rows exposed by the fake cursor.
        resolved_ips (List[str]): IP addresses returned by the resolver fixture.
        recorder (Optional[Dict[str, Any]]): Existing recorder to populate, or None.

    Yields:
        Dict[str, Any]: Captured connection, query, fetch and cleanup calls.
    """
    recorder = recorder if recorder is not None else {}

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
    ) as resolver, mock.patch.object(
        v1, "install_pinned_transport"
    ) as install, mock.patch.object(
        v1, "connector_available", return_value=True
    ), mock.patch.object(
        DatabricksSQLImageSourceBlockV1,
        "run",
        DatabricksSQLImageSourceBlockV1._run_in_process,
    ):
        recorder["resolver"] = resolver
        recorder["install_pinned_transport"] = install
        yield recorder


def encode(image: np.ndarray, extension: str) -> bytes:
    encoded, buffer = cv2.imencode(extension, image)
    assert encoded
    return buffer.tobytes()


def jpeg_bytes(height: int = 32, width: int = 48, value: int = 200) -> bytes:
    return encode(np.full((height, width, 3), value, dtype=np.uint8), ".jpg")


def pil_bytes(image_format: str, size=(48, 32)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (10, 20, 30)).save(buffer, format=image_format)
    return buffer.getvalue()


def run_block(**overrides: Any) -> list:
    """Run the image block with fixture defaults and explicit input overrides.

    Args:
        **overrides (Any): Block inputs replacing the fixture defaults.

    Returns:
        list: Image, name and metadata rows emitted by the block.
    """
    parameters = {
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
        "access_token": "dapi-secret",
        "catalog": "main",
        "schema_name": "vision",
        "table": "images",
        "metadata_columns": ["id"],
        "image_column": "bytes",
        "image_name_column": "id",
        "max_rows": 100,
        "timeout_seconds": 30,
        "skip_undecodable": True,
    }
    parameters.update(overrides)
    rows = DatabricksSQLImageSourceBlockV1()._run_in_process(**parameters)

    return rows


# ---------------------------------------------------------------------------
# Manifest validation (compile time)
# ---------------------------------------------------------------------------


def manifest(**overrides: Any) -> dict:
    """Build a workflow manifest using fixture defaults.

    Args:
        **overrides (Any): Manifest fields replacing the fixture defaults.

    Returns:
        dict: The raw manifest to submit to schema validation.
    """
    raw = {
        "type": "roboflow_core/databricks_sql_image_source@v1",
        "name": "db",
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
        "access_token": "$inputs.databricks_token",
        "catalog": "main",
        "schema_name": "vision",
        "table": "images",
        "metadata_columns": ["id"],
    }
    raw.update(overrides)

    return raw


def test_manifest_parsing_when_input_is_valid() -> None:
    result = BlockManifest.model_validate(
        manifest(
            server_hostname="$inputs.databricks_host",
            image_name_column="id",
            max_rows=5,
        )
    )
    assert result.server_hostname == "$inputs.databricks_host"
    assert result.access_token == "$inputs.databricks_token"
    assert result.image_column == "bytes"
    assert result.max_rows == 5
    assert result.timeout_seconds == 120
    assert result.skip_undecodable is True


@pytest.mark.parametrize("max_rows", [0, 1_001, 10_000])
def test_manifest_rejects_max_rows_out_of_range(max_rows: int) -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(manifest(max_rows=max_rows))


@pytest.mark.parametrize("timeout_seconds", [0, 4, 601])
def test_manifest_rejects_timeout_out_of_range(timeout_seconds: int) -> None:
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(manifest(timeout_seconds=timeout_seconds))


def test_manifest_accepts_token_as_selector_or_literal() -> None:
    # The engine re-validates the field with the substituted runtime value, so
    # the type must admit a plain string; the docs steer users to inputs.
    assert (
        BlockManifest.model_validate(manifest(access_token="dapi-literal")).access_token
        == "dapi-literal"
    )
    assert (
        BlockManifest.model_validate(
            manifest(access_token="$inputs.databricks_token")
        ).access_token
        == "$inputs.databricks_token"
    )


@pytest.mark.parametrize(
    "host", ["10.0.0.5", "evil.com", "https://x.cloud.databricks.com"]
)
def test_manifest_rejects_literal_non_databricks_host_at_compile_time(
    host: str,
) -> None:
    with pytest.raises(ValidationError, match="Databricks"):
        BlockManifest.model_validate(manifest(server_hostname=host))


@pytest.mark.parametrize("path", ["/anything", "/sql/1.0/warehouses/abc\n"])
def test_manifest_rejects_literal_bad_path_at_compile_time(path: str) -> None:
    with pytest.raises(ValidationError, match="http_path"):
        BlockManifest.model_validate(manifest(http_path=path))


def test_manifest_declares_outputs_restrictions_and_nesting() -> None:
    """Verify manifest declares outputs restrictions and nesting."""
    assert BlockManifest.get_output_dimensionality_offset() == 1
    assert {o.name for o in BlockManifest.describe_outputs()} == {
        "images",
        "image_names",
        "metadata",
    }
    codes = {r.code for r in BlockManifest.get_restrictions()}
    assert codes == {"queries_warehouse_on_every_run"}
    with mock.patch.object(v1, "block_enabled", return_value=False):
        assert {r.code for r in BlockManifest.get_restrictions()} == {
            "unavailable_on_hosted_platform",
            "queries_warehouse_on_every_run",
        }
    parsed = BlockManifest.model_validate(manifest())
    discovery = parsed.get_actual_restrictions(ignore_environment_restrictions=True)
    assert {r.code for r in discovery.items} == codes
    assert discovery.complete


# ---------------------------------------------------------------------------
# Destination validation (SSRF guard)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "host, expected",
    [
        (
            "dbc-a1b2c3d4-e5f6.cloud.databricks.com",
            "dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        ),
        (
            "DBC-A1B2C3D4-E5F6.Cloud.Databricks.Com.",
            "dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        ),
        (
            "adb-1234567890123456.7.azuredatabricks.net",
            "adb-1234567890123456.7.azuredatabricks.net",
        ),
        (
            "1234567890123456.7.gcp.databricks.com",
            "1234567890123456.7.gcp.databricks.com",
        ),
    ],
)
def test_normalise_hostname_accepts_databricks_hosts(host: str, expected: str) -> None:
    assert normalise_hostname(host) == expected


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
        "evilazuredatabricks.net",
        "evilcloud.databricks.com",
        "https://dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com/path",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com:8443",
        "user@dbc-a1b2c3d4-e5f6.cloud.databricks.com",
        "dbc-a1b2c3d4-e5f6.cloud.databricks.com\\evil",
        "dbc-a.cloud.databricks.com\n",
        " dbc-a.cloud.databricks.com",
        "dbc-a.cloud.databricks.com ",
        "dbc-a.cloud.databricks.com\t",
        "",
        None,
    ],
)
def test_normalise_hostname_rejects_non_databricks_hosts(host: Any) -> None:
    with pytest.raises(ValueError):
        normalise_hostname(host)


@pytest.mark.parametrize(
    "http_path, kind",
    [
        ("/sql/1.0/warehouses/abcdef1234567890", "warehouse"),
        ("/sql/1.0/endpoints/abcdef1234567890", "warehouse"),
        ("/sql/protocolv1/o/1234567890123456/0123-456789-abcdefg1", "cluster"),
    ],
)
def test_classify_http_path_accepts_warehouse_and_cluster_paths(
    http_path: str, kind: str
) -> None:
    assert classify_http_path(http_path) == kind


@pytest.mark.parametrize(
    "http_path",
    [
        "/anything",
        "/sql/1.0/warehouses/../../admin",
        "sql/1.0/warehouses/abc",
        "",
        "/sql/1.0/warehouses/abc?x=1",
        "/sql/1.0/warehouses/abc\n",
        None,
    ],
)
def test_classify_http_path_rejects_other_paths(http_path: Any) -> None:
    with pytest.raises(ValueError, match="http_path"):
        classify_http_path(http_path)


def test_validate_destination_pins_the_first_global_answer() -> None:
    with mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=["52.1.2.3", "52.1.2.4"]
    ) as resolver:
        destination = validate_destination(HOST.upper() + ".", http_path=HTTP_PATH)
    assert destination.hostname == HOST
    assert destination.ip_address == "52.1.2.3"
    assert destination.port == 443
    assert resolver.call_args.kwargs == {
        "host": HOST,
        "port": 443,
        "allow_non_global_addresses": False,
    }


def test_validate_destination_never_connects_before_syntax_checks() -> None:
    with mock.patch.object(v1, "resolve_and_validate_ips") as resolver:
        with pytest.raises(ValueError):
            validate_destination("internal.corp", http_path=HTTP_PATH)
        with pytest.raises(ValueError, match="http_path"):
            validate_destination(HOST, http_path="/anything")
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


def test_non_global_addresses_are_denied_by_default_and_opt_in_self_hosted_only() -> (
    None
):
    with mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=[PUBLIC_IP]
    ) as resolver:
        validate_destination(HOST, http_path=HTTP_PATH)
        assert resolver.call_args.kwargs["allow_non_global_addresses"] is False
    with mock.patch.object(
        v1, "ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES", True
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=["10.0.0.5"]
    ) as resolver:
        destination = validate_destination(HOST, http_path=HTTP_PATH)
        assert resolver.call_args.kwargs["allow_non_global_addresses"] is True
        assert destination.ip_address == "10.0.0.5"


@pytest.mark.parametrize("flag", ["GCP_SERVERLESS", "LAMBDA"])
def test_hosted_platform_forces_global_addresses_and_refuses_cluster_paths(
    flag: str,
) -> None:
    hosted = {"GCP_SERVERLESS": False, "LAMBDA": False, flag: True}
    with mock.patch.object(
        v1, "running_on_hosted_platform", return_value=True
    ), mock.patch.object(
        v1, "ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES", True
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=[PUBLIC_IP]
    ) as resolver:
        validate_destination(HOST, http_path=HTTP_PATH)
        assert resolver.call_args.kwargs["allow_non_global_addresses"] is False, hosted
        with pytest.raises(ValueError, match="cluster paths"):
            validate_destination(
                HOST, http_path="/sql/protocolv1/o/123/0123-456789-abcdefg1"
            )


def test_run_validates_destination_before_connecting() -> None:
    with fake_databricks([{"id": 1, "bytes": jpeg_bytes()}]) as recorder:
        with pytest.raises(ValueError, match="not a Databricks workspace host"):
            run_block(server_hostname="internal.corp")
    assert "connect_kwargs" not in recorder


# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------


def test_run_refuses_when_the_block_is_disabled_before_any_lookup_or_connection() -> (
    None
):
    with fake_databricks(
        [{"id": 1, "bytes": jpeg_bytes()}]
    ) as recorder, mock.patch.object(v1, "block_enabled", return_value=False):
        with pytest.raises(RuntimeError, match="DATABRICKS_SQL_IMAGE_SOURCE_ENABLED"):
            run_block()
    assert "connect_kwargs" not in recorder
    recorder["resolver"].assert_not_called()


def test_run_allowed_on_hosted_platform_when_explicitly_enabled() -> None:
    with fake_databricks(
        [{"id": 1, "bytes": jpeg_bytes()}]
    ) as recorder, mock.patch.object(
        v1, "running_on_hosted_platform", return_value=True
    ), mock.patch.object(
        v1, "block_enabled", return_value=True
    ):
        result = run_block()
    assert len(result) == 1 and "connect_kwargs" in recorder


def test_enablement_is_independent_of_the_connector_being_installed() -> None:
    with mock.patch.object(v1, "block_enabled", return_value=True), mock.patch.object(
        v1, "connector_available", return_value=False
    ):
        with pytest.raises(RuntimeError, match="databricks-sql-connector"):
            run_block()
    with mock.patch.object(v1, "block_enabled", return_value=False), mock.patch.object(
        v1, "connector_available", return_value=True
    ):
        with pytest.raises(RuntimeError, match="DATABRICKS_SQL_IMAGE_SOURCE_ENABLED"):
            run_block()


def test_run_refuses_when_connector_missing() -> None:
    with mock.patch.object(
        v1, "connector_available", return_value=False
    ), mock.patch.object(v1, "resolve_and_validate_ips") as resolver:
        with pytest.raises(RuntimeError, match="databricks-sql-connector"):
            run_block()
    resolver.assert_not_called()


def test_run_honours_disable_sinks_execution_policy() -> None:
    """Verify run honours disable sinks execution policy."""
    with fake_databricks([{"id": 1, "bytes": jpeg_bytes()}]) as recorder:
        block = DatabricksSQLImageSourceBlockV1(disable_sinks=True)
        with pytest.raises(RuntimeError, match="disable_sinks"):
            block.run(
                server_hostname=HOST,
                http_path=HTTP_PATH,
                access_token="t",
                catalog="main",
                schema_name="vision",
                table="images",
                metadata_columns=[],
                image_column="bytes",
                image_name_column=None,
                max_rows=10,
                timeout_seconds=30,
                skip_undecodable=True,
            )
    assert "connect_kwargs" not in recorder
    assert DatabricksSQLImageSourceBlockV1.get_init_parameters() == ["disable_sinks"]


# ---------------------------------------------------------------------------
# Connection parameters, pinning scope, deadline and cleanup
# ---------------------------------------------------------------------------


def test_connection_kwargs_close_every_unpinned_request_path() -> None:
    """Verify connection kwargs close every unpinned request path."""
    destination = v1.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    kwargs = connection_kwargs(destination, http_path=HTTP_PATH, timeout_seconds=90.6)
    assert kwargs == {
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
        "_port": 443,
        "use_cloud_fetch": False,
        "use_inline_params": False,
        "enable_query_result_lz4_compression": False,
        "enable_telemetry": False,
        "session_configuration": {"STATEMENT_TIMEOUT": "90"},
        "_socket_timeout": 30,
        "_retry_stop_after_attempts_count": 3,
        "_retry_stop_after_attempts_duration": 90,
        "_retry_delay_max": 5,
        "_retry_max_redirects": 0,
    }
    assert (
        connection_kwargs(destination, http_path=HTTP_PATH, timeout_seconds=0.2)[
            "_socket_timeout"
        ]
        == 1
    )


def test_run_connects_with_normalised_host_inside_a_pinned_scope() -> None:
    seen = {}

    def connect(**kwargs):
        seen["pin"] = v1.current_pin() if hasattr(v1, "current_pin") else None
        from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
            transport,
        )

        seen["pin"] = transport.current_pin()
        seen["kwargs"] = kwargs
        return FakeConnection([{"id": 1, "bytes": jpeg_bytes()}], seen)

    sql_module = types.ModuleType("databricks.sql")
    sql_module.connect = connect
    databricks_module = types.ModuleType("databricks")
    databricks_module.sql = sql_module
    with mock.patch.dict(
        sys.modules, {"databricks": databricks_module, "databricks.sql": sql_module}
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=[PUBLIC_IP]
    ), mock.patch.object(
        v1, "install_pinned_transport"
    ) as install, mock.patch.object(
        v1, "connector_available", return_value=True
    ):
        result = run_block(server_hostname=HOST.upper() + ".")
    assert len(result) == 1
    install.assert_called_once()
    assert seen["kwargs"]["server_hostname"] == HOST
    assert seen["kwargs"]["access_token"] == "dapi-secret"
    assert seen["pin"].hostname == HOST and seen["pin"].ip_address == PUBLIC_IP
    from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
        transport,
    )

    assert transport.current_pin() is None, "pin must not leak outside the run"


def test_run_bounds_the_connectors_result_buffer_and_row_limit() -> None:
    rows = [{"id": i, "bytes": jpeg_bytes()} for i in range(3)]
    with fake_databricks(rows) as recorder, mock.patch.object(
        v1, "MAX_IMAGE_PAYLOAD_BYTES", 4 * 1024 * 1024
    ):
        run_block(max_rows=2)
    assert recorder["cursor_kwargs"] == {
        "arraysize": v1.FETCH_BATCH_SIZE,
        "buffer_size_bytes": 8 * 1024 * 1024,
        "row_limit": 2,
    }
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", 100):
        assert v1.result_buffer_bytes() == v1.MIN_RESULT_BUFFER_BYTES


def _live_timers() -> list:
    return [t for t in threading.enumerate() if isinstance(t, threading.Timer)]


def test_repeated_failures_leave_no_threads_timers_or_open_connections() -> None:
    assert _live_timers() == []
    for _ in range(5):
        with fake_databricks([{"id": 1, "content": jpeg_bytes()}]) as recorder:
            with pytest.raises(ValueError):
                run_block()
        assert recorder["cursor_closed"] and recorder["connection_closed"]
        with fake_databricks([{"id": 1, "bytes": b"junk"}]) as recorder:
            with pytest.raises(ValueError, match="skipped"):
                run_block()
        assert recorder["cursor_closed"] and recorder["connection_closed"]
    time.sleep(0.05)
    assert _live_timers() == [], "a deadline watchdog outlived its run"


def test_run_times_out_and_cancels_a_slow_query() -> None:
    """Verify run times out and cancels a slow query."""
    rows = [{"id": 1, "bytes": jpeg_bytes()}]
    recorder = {"execute_delay": 1.5}
    with fake_databricks(rows, recorder=recorder), mock.patch.object(
        v1, "MAX_TIMEOUT_SECONDS", 600
    ):
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timeout_seconds"):
            DatabricksSQLImageSourceBlockV1()._run_in_process(
                server_hostname=HOST,
                http_path=HTTP_PATH,
                access_token="t",
                catalog="main",
                schema_name="vision",
                table="images",
                metadata_columns=[],
                image_column="bytes",
                image_name_column=None,
                max_rows=10,
                timeout_seconds=1,
                skip_undecodable=True,
            )
    assert time.monotonic() - started < 5
    assert recorder["cancelled"] is True
    assert recorder["connection_closed"] is True


def test_run_closes_cursor_and_connection_when_the_row_loop_raises() -> None:
    rows = [{"id": 1, "content": jpeg_bytes()}]
    with fake_databricks(rows) as recorder:
        with pytest.raises(ValueError, match="Column `bytes` not present"):
            run_block()
    assert recorder["cursor_closed"] is True
    assert recorder["connection_closed"] is True


def test_run_redacts_the_token_from_connector_errors() -> None:
    def connect(**kwargs):
        raise RuntimeError("401 bad token dapi-secret for host")

    sql_module = types.ModuleType("databricks.sql")
    sql_module.connect = connect
    databricks_module = types.ModuleType("databricks")
    databricks_module.sql = sql_module
    with mock.patch.dict(
        sys.modules, {"databricks": databricks_module, "databricks.sql": sql_module}
    ), mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=[PUBLIC_IP]
    ), mock.patch.object(
        v1, "install_pinned_transport"
    ), mock.patch.object(
        v1, "connector_available", return_value=True
    ):
        with pytest.raises(RuntimeError) as info:
            run_block()
    assert "dapi-secret" not in str(info.value)
    assert "***" in str(info.value)


# ---------------------------------------------------------------------------
# Decoding and limits
# ---------------------------------------------------------------------------


def test_decode_image_accepts_bytes_bytearray_memoryview_and_base64() -> None:
    raw = jpeg_bytes()
    for payload in (
        raw,
        bytearray(raw),
        memoryview(raw),
        base64.b64encode(raw).decode(),
    ):
        assert decode_image(payload).shape == (32, 48, 3)


@pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
def test_decode_image_accepts_allowlisted_formats(image_format: str) -> None:
    assert decode_image(pil_bytes(image_format)).shape == (32, 48, 3)


@pytest.mark.parametrize("image_format", ["TIFF", "BMP", "GIF"])
def test_decode_image_rejects_other_formats_without_decoding(image_format: str) -> None:
    with mock.patch.object(v1.cv2, "imdecode") as imdecode:
        with pytest.raises(ValueError, match="not allowed"):
            decode_image(pil_bytes(image_format))
    imdecode.assert_not_called()


def test_decode_image_rejects_tiff_with_lying_header_before_allocating() -> None:
    # A TIFF whose header dimensions disagree with the pixel data was the known
    # bypass of a header-only cap; the format allowlist refuses it outright.
    """Verify decode image rejects tiff with lying header before allocating."""
    buffer = io.BytesIO()
    Image.new("RGB", (60, 60)).save(buffer, format="TIFF")
    with mock.patch.object(v1.cv2, "imdecode") as imdecode:
        with pytest.raises(ValueError, match="not allowed"):
            decode_image(buffer.getvalue())
    imdecode.assert_not_called()


@pytest.mark.parametrize("payload", [None, b"not an image", 42])
def test_decode_image_rejects_invalid_payloads(payload: Any) -> None:
    with pytest.raises(ValueError):
        decode_image(payload)


def test_decode_image_payload_byte_limit_is_exact() -> None:
    raw = jpeg_bytes()
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", len(raw)):
        decode_image(raw)
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", len(raw) - 1):
        with pytest.raises(ValueError, match="exceeds"):
            decode_image(raw)


def test_decode_image_rejects_base64_payload_over_limit_before_decoding() -> None:
    encoded = base64.b64encode(jpeg_bytes()).decode()
    with mock.patch.object(v1, "MAX_IMAGE_PAYLOAD_BYTES", 100), mock.patch.object(
        v1, "base64"
    ) as b64:
        with pytest.raises(ValueError, match="exceeds 100 bytes"):
            decode_image(encoded)
    b64.b64decode.assert_not_called()


def test_decode_image_pixel_limit_is_exact_and_checked_before_decoding() -> None:
    raw = jpeg_bytes(height=32, width=48)
    with mock.patch.object(v1, "MAX_IMAGE_PIXELS", 32 * 48):
        decode_image(raw)
    with mock.patch.object(v1, "MAX_IMAGE_PIXELS", 32 * 48 - 1), mock.patch.object(
        v1.cv2, "imdecode"
    ) as imdecode:
        with pytest.raises(ValueError, match="48x32"):
            decode_image(raw)
    imdecode.assert_not_called()


def test_decode_image_rejects_decoded_array_larger_than_header() -> None:
    raw = jpeg_bytes(height=32, width=48)
    with mock.patch.object(
        v1.cv2, "imdecode", return_value=np.zeros((64, 64, 3), dtype=np.uint8)
    ):
        with pytest.raises(ValueError, match="does not match its header"):
            decode_image(raw)


def test_decode_image_reserves_encoded_and_decoded_bytes_from_the_budget() -> None:
    raw = jpeg_bytes(height=32, width=48)
    budget = RunBudget(limit=len(raw) + 32 * 48 * 3)
    decode_image(raw, budget=budget)
    assert budget.used == len(raw) + 32 * 48 * 3
    with pytest.raises(v1.RunBudgetExceeded):
        decode_image(raw, budget=budget)


def test_run_budget_covers_metadata_and_names() -> None:
    """Verify run budget covers metadata and names."""
    rows = [{"id": "x" * 200, "bytes": jpeg_bytes(), "note": "y" * 3000}]
    with fake_databricks(rows), mock.patch.object(v1, "MAX_RUN_BYTES", 3_000):
        with pytest.raises(ValueError, match="byte budget"):
            run_block(metadata_columns=["id", "note"])


def test_run_budget_failure_is_not_skippable() -> None:
    rows = [{"id": i, "bytes": jpeg_bytes()} for i in range(3)]
    one_row = len(jpeg_bytes()) + 32 * 48 * 3 + 20
    with fake_databricks(rows), mock.patch.object(v1, "MAX_RUN_BYTES", one_row * 2):
        with pytest.raises(ValueError, match="byte budget"):
            run_block(skip_undecodable=True)


def test_metadata_strings_are_truncated_and_rows_capped() -> None:
    row = {"id": 1, "bytes": b"", "note": "n" * 10_000, "blob": b"\x00"}
    metadata = build_metadata(row, image_column="bytes")
    assert len(metadata["note"]) == v1.MAX_METADATA_VALUE_CHARS
    assert "blob" not in metadata and "bytes" not in metadata
    with mock.patch.object(v1, "MAX_METADATA_BYTES_PER_ROW", 100):
        with pytest.raises(ValueError, match="metadata"):
            build_metadata(row, image_column="bytes")


def test_nested_metadata_is_capped_in_depth_and_size() -> None:
    """Verify nested metadata is capped in depth and size."""
    deep = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
    converted = to_json_safe(deep)
    assert converted["a"]["b"]["c"]["d"] == "{'e': {'f': 1}}"
    wide = {"items": list(range(10_000))}
    assert len(to_json_safe(wide)["items"]) == v1.MAX_METADATA_ITEMS
    many_keys = {f"k{i}": i for i in range(10_000)}
    assert len(to_json_safe(many_keys)) == v1.MAX_METADATA_ITEMS
    # A tiny image with a large metadata column counts against the run budget.
    rows = [{"id": 1, "bytes": jpeg_bytes(), "note": "y" * 60_000}]
    with fake_databricks(rows), mock.patch.object(v1, "MAX_RUN_BYTES", 6_000):
        with pytest.raises(ValueError, match="byte budget"):
            run_block(metadata_columns=["id", "note"])


def test_image_names_are_length_capped() -> None:
    with pytest.raises(ValueError, match="image name"):
        v1.resolve_image_name({"id": "n" * 300}, image_name_column="id")
    assert v1.resolve_image_name({"id": None}, image_name_column="id").count("-") == 4


def test_to_json_safe_keeps_decimal_exact_and_stringifies_unknown_types() -> None:
    assert (
        to_json_safe(decimal.Decimal("12345678901234567890.123"))
        == "12345678901234567890.123"
    )
    assert to_json_safe(datetime.date(2026, 11, 3)) == "2026-11-03"
    assert to_json_safe({"a": [decimal.Decimal(2), b"x"], "b": b"y"}) == {"a": ["2"]}
    assert to_json_safe(datetime.timedelta(seconds=5)) == "0:00:05"
    assert to_json_safe({1, 2}) in ("{1, 2}", "{2, 1}")
    assert to_json_safe(np.int64(3)) == "3"
    assert to_json_safe(True) is True and to_json_safe(None) is None


# ---------------------------------------------------------------------------
# Row handling
# ---------------------------------------------------------------------------


def test_run_emits_one_element_per_row_and_strips_binary_columns() -> None:
    """Verify run emits one element per row and strips binary columns."""
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
    with fake_databricks(rows) as recorder:
        result = run_block(metadata_columns=["id", "raccoon_name", "weight", "thumb"])
    assert [element["image_names"] for element in result] == ["1", "2"]
    assert result[0]["metadata"] == {"id": 1, "raccoon_name": "rocky", "weight": 4.2}
    assert isinstance(result[0]["images"], WorkflowImageData)
    assert result[0]["images"].numpy_image.shape == (32, 48, 3)
    assert "FROM IDENTIFIER(:source) LIMIT :row_limit" in recorder["statement"]
    assert recorder["parameters"]["source"] == "`main`.`vision`.`images`"
    assert recorder["connection_closed"] and recorder["cursor_closed"]


def test_run_fetches_in_batches_and_stops_at_max_rows() -> None:
    rows = [{"id": i, "bytes": jpeg_bytes()} for i in range(10)]
    with fake_databricks(rows) as recorder, mock.patch.object(
        v1, "FETCH_BATCH_SIZE", 4
    ):
        result = run_block(max_rows=7)
    assert [element["image_names"] for element in result] == [str(i) for i in range(7)]
    assert recorder["fetchmany_sizes"] == [4, 3]


def test_run_skips_undecodable_and_oversized_rows_but_fails_when_all_are_skipped() -> (
    None
):
    rows = [
        {"id": 1, "bytes": b"junk"},
        {"id": 2, "bytes": jpeg_bytes()},
        {"id": 3, "bytes": jpeg_bytes(64, 64)},
    ]
    with fake_databricks(rows), mock.patch.object(v1, "MAX_IMAGE_PIXELS", 2_000):
        result = run_block()
    assert [element["image_names"] for element in result] == ["2"]
    with fake_databricks([{"id": 1, "bytes": b"junk"}, {"id": 2, "bytes": b"junk"}]):
        with pytest.raises(ValueError, match="All 2 fetched rows were skipped"):
            run_block()


def test_run_raises_on_undecodable_row_when_skipping_disabled() -> None:
    with fake_databricks(
        [{"id": 1, "bytes": b"junk"}, {"id": 2, "bytes": jpeg_bytes()}]
    ):
        with pytest.raises(ValueError, match="row 0"):
            run_block(skip_undecodable=False)


def test_run_fails_clearly_when_a_configured_column_is_missing() -> None:
    with fake_databricks([{"id": 1, "content": jpeg_bytes()}]):
        with pytest.raises(ValueError, match="Column `bytes` not present"):
            run_block()
    with fake_databricks([{"idd": 1, "bytes": jpeg_bytes()}]):
        with pytest.raises(ValueError, match="image_name_column"):
            run_block(image_name_column="id")


def test_run_returns_empty_list_when_the_query_returns_no_rows() -> None:
    with fake_databricks([]):
        assert run_block() == []


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
            "catalog": "main",
            "schema_name": "vision",
            "table": "images",
            "metadata_columns": ["id", "raccoon_name"],
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
    """Verify engine runs block as input free step producing a batch."""
    rows = [
        {"id": i, "bytes": jpeg_bytes(), "raccoon_name": n}
        for i, n in zip((7, 8, 9), ("rocky", "bandit", "meeko"))
    ]
    execution_engine = ExecutionEngine.init(
        workflow_definition=SOURCE_WORKFLOW,
        init_parameters={"workflows_core.step_execution_mode": StepExecutionMode.LOCAL},
        max_concurrent_steps=1,
    )
    with fake_databricks(rows) as recorder:
        result = execution_engine.run(
            runtime_parameters={"databricks_token": "dapi-xyz", "sql": "SELECT 1"}
        )
    assert recorder["connect_kwargs"]["access_token"] == "dapi-xyz"
    assert recorder["parameters"]["row_limit"] == 5
    assert len(result) == 1
    assert result[0]["names"] == ["7", "8", "9"]
    assert [m["raccoon_name"] for m in result[0]["metadata"]] == [
        "rocky",
        "bandit",
        "meeko",
    ]
    assert all(isinstance(image, WorkflowImageData) for image in result[0]["images"])


def test_engine_returns_empty_lists_when_query_yields_no_rows() -> None:
    execution_engine = ExecutionEngine.init(
        workflow_definition=SOURCE_WORKFLOW,
        init_parameters={"workflows_core.step_execution_mode": StepExecutionMode.LOCAL},
        max_concurrent_steps=1,
    )
    with fake_databricks([]):
        result = execution_engine.run(
            runtime_parameters={"databricks_token": "dapi-xyz", "sql": "SELECT 1"}
        )
    assert result[0] == {"images": [], "names": [], "metadata": []}


def test_engine_rejects_a_submitted_workflow_while_disabled_without_connecting(
    monkeypatch,
) -> None:
    from roboflow_workflows.core_steps import loader
    from roboflow_workflows.core_steps.integrations.databricks import availability
    from roboflow_workflows.errors import WorkflowDefinitionError
    from roboflow_workflows.execution_engine.introspection import blocks_loader

    monkeypatch.setattr(availability, "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED", False)
    monkeypatch.setattr(v1, "block_enabled", lambda: False)
    blocks_loader.clear_caches()
    # The parser memoises parsed definitions per process; availability cannot
    # change within a process (the flag is read at startup), so an unseen
    # definition is what a freshly started disabled worker would compile.
    definition = json.loads(json.dumps(SOURCE_WORKFLOW))
    definition["steps"][0]["name"] = "db_while_disabled"
    for output in definition["outputs"]:
        output["selector"] = output["selector"].replace(
            "$steps.db.", "$steps.db_while_disabled."
        )
    assert loader.load_unavailable_blocks() == {
        "roboflow_core/databricks_sql_image_source@v1": availability.DISABLED_HINT
    }
    assert all(
        b.__name__ != "DatabricksSQLImageSourceBlockV1" for b in loader.load_blocks()
    )
    try:
        with fake_databricks([{"id": 1, "bytes": jpeg_bytes()}]) as recorder:
            with pytest.raises(WorkflowDefinitionError, match="not available"):
                ExecutionEngine.init(
                    workflow_definition=definition,
                    init_parameters={
                        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL
                    },
                    max_concurrent_steps=1,
                )
        assert "connect_kwargs" not in recorder
        recorder["resolver"].assert_not_called()
    finally:
        blocks_loader.clear_caches()
