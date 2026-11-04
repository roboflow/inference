"""Network behaviour of the block with the real connector and pinned transport.

These tests run ``roboflow_core/databricks_sql_image_source@v1`` unchanged,
with the real ``databricks-sql-connector``, against the local HTTPS Thrift
warehouse in ``databricks_thrift_harness.py``. Destination validation is the
only thing patched (the test host resolves to 127.0.0.1 on the harness port),
and a socket-level guard records every connection attempt the process makes
and refuses anything that is not the harness, so a forbidden destination is
asserted from the attempt, never from a request that may already have left.

Covered: successful import across several fetch chunks, SNI / ``Host`` /
single-destination invariants, no telemetry or feature-flag traffic, DNS
changing after validation, retryable errors and forced reconnects, redirects
to forbidden destinations, warehouse-supplied result links, invalid
certificates, proxies, slow-query cancellation with bounded cleanup, and
recovery after failures.
"""

import os
import socket
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List
from unittest import mock

import cv2
import numpy as np
import pytest
import urllib3.util.connection
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    transport,
    v1,
)
from roboflow_workflows.execution_engine.core import ExecutionEngine

pytest.importorskip("databricks.sql", reason="databricks extra not installed")
pytest.importorskip("thrift", reason="databricks extra not installed")

from tests.unit_tests.core_steps.integrations import (  # noqa: E402
    databricks_thrift_harness as harness,
)

HOST = harness.HOST
HTTP_PATH = harness.HTTP_PATH
METADATA_HOST = "169.254.169.254"
LINK_HOST = "result-links.example.invalid"
POISONED_ANSWER = "10.0.0.5"


def block_threads() -> List[str]:
    """Names of live threads other than the harness server's own."""
    return sorted(
        thread.name
        for thread in threading.enumerate()
        if "serve_forever" not in thread.name
        and "process_request_thread" not in thread.name
    )


def jpeg(value: int = 200) -> bytes:
    encoded, buffer = cv2.imencode(".jpg", np.full((32, 48, 3), value, np.uint8))
    assert encoded
    return buffer.tobytes()


def image_rows(count: int) -> List[Dict[str, Any]]:
    return [{"id": i, "bytes": jpeg((i * 7) % 255)} for i in range(count)]


@contextmanager
def network_guard(allowed_port: int) -> Iterator[Dict[str, List[Any]]]:
    """Record every connection attempt and DNS lookup; allow only the harness."""
    seen: Dict[str, List[Any]] = {"connections": [], "lookups": []}
    real_create_connection = urllib3.util.connection.create_connection
    real_socket_create_connection = socket.create_connection
    real_getaddrinfo = socket.getaddrinfo

    def guard(address, *args, **kwargs):
        seen["connections"].append(tuple(address))
        if address[0] != "127.0.0.1" or int(address[1]) != allowed_port:
            raise OSError(f"connection to {address} refused by the test guard")
        return real_create_connection(address, *args, **kwargs)

    def socket_guard(address, *args, **kwargs):
        seen["connections"].append(tuple(address))
        if address[0] != "127.0.0.1" or int(address[1]) != allowed_port:
            raise OSError(f"connection to {address} refused by the test guard")
        return real_socket_create_connection(address, *args, **kwargs)

    def getaddrinfo(host, port, *args, **kwargs):
        seen["lookups"].append(host)
        if host == HOST:
            # DNS "changed" after validation: the name now answers a private
            # address. The pinned transport must never ask.
            return [
                (socket.AF_INET, socket.SOCK_STREAM, 6, "", (POISONED_ANSWER, port))
            ]
        return real_getaddrinfo(host, port, *args, **kwargs)

    with mock.patch.object(
        urllib3.util.connection, "create_connection", side_effect=guard
    ), mock.patch.object(
        socket, "create_connection", side_effect=socket_guard
    ), mock.patch.object(
        socket, "getaddrinfo", side_effect=getaddrinfo
    ):
        yield seen


@contextmanager
def warehouse(
    behaviour: harness.Behaviour, certificate_host: str = HOST
) -> Iterator[harness.FakeWarehouse]:
    """Serve ``behaviour`` and point the block's validated destination at it."""
    with harness.FakeWarehouse(
        behaviour, certificate_host=certificate_host
    ) as server, mock.patch.object(
        v1, "resolve_and_validate_ips", return_value=["127.0.0.1"]
    ), mock.patch.object(
        v1, "ALLOW_DATABRICKS_WORKFLOWS_SOURCE_TO_NON_GLOBAL_ADDRESSES", True
    ), mock.patch.object(
        v1, "DATABRICKS_HTTPS_PORT", server.port
    ), mock.patch.object(
        v1, "TLS_TRUSTED_CA_FILE", harness.CA_FILE
    ), mock.patch.object(
        v1, "block_enabled", return_value=True
    ):
        yield server


def run_block(timeout_seconds: int = 30, **overrides: Any) -> list:
    parameters = {
        "server_hostname": HOST,
        "http_path": HTTP_PATH,
        "access_token": "dapi-test-token",
        "sql_statement": "SELECT id, bytes FROM catalog.schema.images",
        "image_column": "bytes",
        "image_name_column": "id",
        "max_rows": 100,
        "timeout_seconds": timeout_seconds,
        "skip_undecodable": True,
    }
    parameters.update(overrides)
    return v1.DatabricksSQLImageSourceBlockV1().run(**parameters)


def assert_only_the_pinned_warehouse_was_contacted(
    server: harness.FakeWarehouse, seen: Dict[str, List[Any]]
) -> None:
    assert seen["connections"], "the connector made no connection at all"
    assert all(
        address == ("127.0.0.1", server.port) for address in seen["connections"]
    ), seen["connections"]
    assert HOST not in seen["lookups"], "the pinned transport resolved the name"
    assert all(
        connection["sni"] == HOST and connection["peer"] == "127.0.0.1"
        for connection in server.observations.connections
    ), server.observations.connections
    assert {r["host_header"] for r in server.observations.requests} <= {HOST}
    assert {r["path"] for r in server.observations.requests} <= {
        HTTP_PATH
    }, "a request left the warehouse path (telemetry / feature flags / staging)"
    assert transport.current_pin() is None, "pin leaked out of the run"


# --------------------------------------------------------------- happy path


def test_imports_rows_across_fetch_chunks_over_the_pinned_tls_connection() -> None:
    behaviour = harness.Behaviour(rows=image_rows(20), chunk_rows=8)
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        result = run_block()
    assert [element["image_names"] for element in result] == [str(i) for i in range(20)]
    assert all(element["images"].numpy_image.shape == (32, 48, 3) for element in result)
    assert [element["metadata"] for element in result][:2] == [{"id": 0}, {"id": 1}]
    methods = server.observations.thrift_methods
    assert methods.count("FetchResults") == 3, methods
    assert methods[-2:] == ["CloseOperation", "CloseSession"], "cleanup on success"
    assert_only_the_pinned_warehouse_was_contacted(server, seen)


def test_runs_through_the_execution_engine_with_the_real_connector() -> None:
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "token"}],
        "steps": [
            {
                "type": "roboflow_core/databricks_sql_image_source@v1",
                "name": "db",
                "server_hostname": HOST,
                "http_path": HTTP_PATH,
                "access_token": "$inputs.token",
                "sql_statement": "SELECT id, bytes FROM t",
                "image_name_column": "id",
                "max_rows": 3,
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "names", "selector": "$steps.db.image_names"}
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={"workflows_core.step_execution_mode": StepExecutionMode.LOCAL},
        max_concurrent_steps=1,
    )
    behaviour = harness.Behaviour(rows=image_rows(5), chunk_rows=2)
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        result = engine.run(runtime_parameters={"token": "dapi-test-token"})
    assert result[0]["names"] == ["0", "1", "2"]
    assert_only_the_pinned_warehouse_was_contacted(server, seen)


# ---------------------------------------------------- destination invariants


def test_dns_changing_after_validation_cannot_steer_the_connection() -> None:
    # `network_guard` answers the hostname with a private address; the run
    # still lands on the validated address and never asks the resolver.
    behaviour = harness.Behaviour(rows=image_rows(2))
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        result = run_block()
    assert len(result) == 2
    assert HOST not in seen["lookups"]
    assert_only_the_pinned_warehouse_was_contacted(server, seen)


def test_retryable_error_and_forced_reconnect_stay_on_the_pinned_address() -> None:
    behaviour = harness.Behaviour(
        rows=image_rows(3),
        http_script=[
            harness.HttpAction(status=503, headers={"Retry-After": "1"}),
            harness.HttpAction(close_connection=True),
        ],
    )
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        result = run_block()
    assert len(result) == 3
    assert len(seen["connections"]) >= 2, "the connector reconnected"
    assert_only_the_pinned_warehouse_was_contacted(server, seen)


@pytest.mark.parametrize(
    "location",
    [
        f"http://{METADATA_HOST}/latest/meta-data/",
        f"https://{METADATA_HOST}/latest/meta-data/",
        f"https://{LINK_HOST}{HTTP_PATH}",
        f"http://{HOST}{HTTP_PATH}",
    ],
)
def test_redirects_are_not_followed_to_any_destination(location: str) -> None:
    behaviour = harness.Behaviour(
        rows=image_rows(1),
        http_script=[harness.HttpAction(status=302, headers={"Location": location})],
    )
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        with pytest.raises(Exception) as info:
            run_block(timeout_seconds=10)
    assert "dapi-test-token" not in str(info.value)
    assert all(
        address == ("127.0.0.1", server.port) for address in seen["connections"]
    ), seen["connections"]
    assert METADATA_HOST not in seen["lookups"] and LINK_HOST not in seen["lookups"]
    assert HOST not in seen["lookups"], "a redirect back to the name re-resolves"


def test_result_links_supplied_by_the_warehouse_are_never_downloaded() -> None:
    behaviour = harness.Behaviour(
        rows=image_rows(1), result_links=[f"https://{LINK_HOST}/chunk-0.arrow"]
    )
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        with pytest.raises(Exception) as info:
            run_block(timeout_seconds=10)
    assert "dapi-test-token" not in str(info.value)
    assert "result links" in str(info.value)
    assert LINK_HOST not in seen["lookups"]
    assert all(
        address == ("127.0.0.1", server.port) for address in seen["connections"]
    ), seen["connections"]
    assert transport.current_pin() is None
    assert not transport.pin_active_anywhere()


def test_invalid_certificate_is_refused_before_any_request_is_sent() -> None:
    behaviour = harness.Behaviour(rows=image_rows(1))
    with warehouse(behaviour, certificate_host=harness.OTHER_HOST) as server:
        with network_guard(server.port) as seen:
            with pytest.raises(Exception) as info:
                run_block(timeout_seconds=10)
    assert "dapi-test-token" not in str(info.value)
    assert server.observations.requests == [], "no request may follow a bad cert"
    assert seen["connections"], "the handshake was attempted against the pin"


def test_proxy_settings_are_refused_while_pinned_without_any_connection() -> None:
    behaviour = harness.Behaviour(rows=image_rows(1))
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        with mock.patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:1"}):
            with pytest.raises(RuntimeError, match="proxy"):
                run_block(timeout_seconds=10)
    assert seen["connections"] == []
    assert server.observations.requests == []


# ------------------------------------------------- deadline and recovery


def test_slow_query_is_cancelled_at_the_deadline_and_the_slot_is_released() -> None:
    behaviour = harness.Behaviour(rows=image_rows(2), hold_running=True)
    threads_before = block_threads()
    with warehouse(behaviour) as server, network_guard(server.port) as seen:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="timeout_seconds"):
            run_block(timeout_seconds=5)
        elapsed = time.monotonic() - started
        assert server.observations.cancelled.wait(timeout=v1.CLEANUP_GRACE_SECONDS)
        assert server.observations.session_closed.wait(timeout=v1.CLEANUP_GRACE_SECONDS)
        assert "CANCELED" in server.observations.operation_states
        assert elapsed < 5 + v1.CLEANUP_GRACE_SECONDS, elapsed
        assert_only_the_pinned_warehouse_was_contacted(server, seen)
        # Recovery on the same warehouse: the next run is unaffected.
        behaviour.hold_running = False
        result = run_block(timeout_seconds=10)
        assert len(result) == 2
    time.sleep(0.1)
    assert block_threads() == threads_before, "watchdog / connector threads"


def test_repeated_failures_do_not_accumulate_queries_sessions_or_threads() -> None:
    threads_before = block_threads()
    behaviour = harness.Behaviour(rows=[{"id": 1, "bytes": b"not an image"}])
    with warehouse(behaviour) as server, network_guard(server.port):
        for _ in range(4):
            with pytest.raises(ValueError, match="skipped"):
                run_block(timeout_seconds=10)
        methods = server.observations.thrift_methods
        assert methods.count("CloseSession") == 4
        assert methods.count("CloseOperation") == 4
        behaviour.rows[:] = image_rows(1)
        assert len(run_block(timeout_seconds=10)) == 1
    time.sleep(0.1)
    assert block_threads() == threads_before
