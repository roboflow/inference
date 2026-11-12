"""Transport pinning, availability gating and explicit rejection for Databricks.

The pinning tests use the real ``databricks-sql-connector`` transport class
when the package is installed (it is in the workflows test extras); they open
no network connection, they inspect the pool the transport would use.
"""

import socket
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest import mock

import pytest
from roboflow_workflows.core_steps.integrations.databricks import availability
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    transport,
)
from roboflow_workflows.errors import WorkflowDefinitionError

HOST = "dbc-a1b2c3d4-e5f6.cloud.databricks.com"
PUBLIC_IP = "52.1.2.3"
PRIVATE_IP = "10.0.0.5"

connector = pytest.importorskip(
    "databricks.sql", reason="databricks extra not installed"
)


def _client(hostname: str = HOST):
    from databricks.sql.auth import thrift_http_client
    from databricks.sql.types import SSLOptions

    transport.install_pinned_transport()
    client = thrift_http_client.THttpClient(
        auth_provider=mock.MagicMock(),
        uri_or_host=f"https://{hostname}:443/sql/1.0/warehouses/abc",
        ssl_options=SSLOptions(tls_verify=True),
    )
    return client


def _pool_of(client) -> Any:
    for name, value in vars(client).items():
        if name.endswith("__pool"):
            return value
    raise AssertionError("transport has no pool")


# ------------------------------------------------------------------ pinning


def test_installed_transport_is_the_class_the_backend_instantiates() -> None:
    cls = transport.install_pinned_transport()
    from databricks.sql.auth import thrift_http_client
    from databricks.sql.backend import thrift_backend

    assert thrift_http_client.THttpClient is cls
    assert thrift_backend.databricks.sql.auth.thrift_http_client.THttpClient is cls
    assert issubclass(cls, transport._PinnedMixin)
    assert transport.install_pinned_transport() is cls, "idempotent"


def test_pinned_pool_preserves_hostname_for_tls_and_host() -> None:
    """Verify pinned pool preserves hostname for tls and host."""
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        client = _client()
        client.open()
        client.setCustomHeaders({"User-Agent": "test"})
        pool = _pool_of(client)
    assert pool.host == PUBLIC_IP and pool.port == 443
    assert pool.conn_kw["server_hostname"] == HOST
    assert pool.assert_hostname == HOST, "urllib3 2.x keeps it on the pool"
    assert pool.cert_reqs == __import__("ssl").CERT_REQUIRED
    headers = getattr(client, "_THttpClient__custom_headers")
    assert headers["Host"] == HOST and headers["User-Agent"] == "test"


def test_dns_change_after_validation_cannot_steer_the_connection() -> None:
    # Validation saw a public answer; by connect time DNS answers private.
    """Verify dns change after validation cannot steer the connection."""
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with mock.patch.object(
        socket,
        "getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PRIVATE_IP, 443))],
    ) as resolver, transport.pinned_destination(pin):
        client = _client()
        client.open()
        pool = _pool_of(client)
    assert (
        pool.host == PUBLIC_IP
    ), "pool must use the validated address, not a fresh lookup"
    resolver.assert_not_called()


def test_transport_refuses_a_target_that_differs_from_the_pin() -> None:
    """Verify transport refuses a target that differs from the pin."""
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        client = _client("other.cloud.databricks.com")
        with pytest.raises(transport.PinnedTransportError, match="does not match"):
            client.open()


def test_transport_refuses_proxies_and_disabled_tls_verification_while_pinned() -> None:
    """Verify transport refuses proxies and disabled tls verification while pinned."""
    from databricks.sql.auth import thrift_http_client
    from databricks.sql.types import SSLOptions

    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        client = _client()
        with mock.patch.object(client, "using_proxy", return_value=True):
            with pytest.raises(transport.PinnedTransportError, match="proxy"):
                client.open()
        insecure = thrift_http_client.THttpClient(
            auth_provider=mock.MagicMock(),
            uri_or_host=f"https://{HOST}:443/sql/1.0/warehouses/abc",
            ssl_options=SSLOptions(tls_verify=False),
        )
        with pytest.raises(transport.PinnedTransportError, match="verification"):
            insecure.open()


def test_cross_host_redirects_are_refused_by_the_pinned_pool() -> None:
    """Verify cross host redirects are refused by the pinned pool."""
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        client = _client()
        client.open()
        pool = _pool_of(client)
    assert pool.is_same_host(f"https://{PUBLIC_IP}:443/x")
    assert not pool.is_same_host("https://169.254.169.254/latest/meta-data")
    assert not pool.is_same_host(
        f"https://{HOST}/x"
    ), "a redirect back to the name re-resolves; refused"


def test_transport_is_unchanged_outside_a_pinned_scope() -> None:
    client = _client()
    assert transport.current_pin() is None
    with mock.patch("urllib3.HTTPSConnectionPool") as pool_cls:
        pool_cls.return_value = mock.MagicMock()
    # No pin: the connector's own open() runs and pools the hostname.
    client.open()
    pool = _pool_of(client)
    assert pool.host == HOST
    assert "server_hostname" not in pool.conn_kw


# ------------------------------------------------------- availability gate


def test_block_is_registered_only_when_connector_and_policy_allow(monkeypatch) -> None:
    from roboflow_workflows.core_steps import loader

    def names():
        return {b.__name__ for b in loader.load_blocks()}

    monkeypatch.setattr(loader, "databricks_unavailability_reason", lambda: None)
    assert "DatabricksSQLImageSourceBlockV1" in names()
    assert loader.load_unavailable_blocks() == {}

    monkeypatch.setattr(
        loader, "databricks_unavailability_reason", lambda: "connector missing"
    )
    assert "DatabricksSQLImageSourceBlockV1" not in names()
    assert loader.load_unavailable_blocks() == {
        "roboflow_core/databricks_sql_image_source@v1": "connector missing"
    }


def test_unavailability_reason_prefers_policy_then_dependency(monkeypatch) -> None:
    monkeypatch.setattr(availability, "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED", False)
    monkeypatch.setattr(availability, "connector_available", lambda: True)
    reason = availability.databricks_unavailability_reason()
    assert "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED" in reason and "restart" in reason
    monkeypatch.setattr(availability, "DATABRICKS_SQL_IMAGE_SOURCE_ENABLED", True)
    monkeypatch.setattr(availability, "connector_available", lambda: False)
    assert "not installed" in availability.databricks_unavailability_reason()
    monkeypatch.setattr(availability, "connector_available", lambda: True)
    assert availability.databricks_unavailability_reason() is None


def test_connector_available_reflects_importability(monkeypatch) -> None:
    assert availability.connector_available() is True
    with mock.patch.dict(sys.modules, {"databricks": None, "databricks.sql": None}):
        assert availability.connector_available() is False


# ---------------------------------------------------- explicit rejection


def test_workflow_referencing_an_unavailable_block_is_rejected_with_the_reason(
    monkeypatch,
    request,
) -> None:
    """Verify workflow referencing an unavailable block is rejected with the reason.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
        request (pytest.FixtureRequest): Pytest request used to isolate fixture teardown.
    """
    from roboflow_workflows.core_steps import loader
    from roboflow_workflows.execution_engine.introspection import blocks_loader
    from roboflow_workflows.execution_engine.v1.compiler.syntactic_parser import (
        parse_workflow_definition,
    )

    monkeypatch.setattr(
        loader,
        "databricks_unavailability_reason",
        lambda: "the `databricks-sql-connector` package is not installed",
    )
    blocks_loader.clear_caches()
    request.addfinalizer(blocks_loader.clear_caches)
    available = [
        b
        for b in blocks_loader.load_workflow_blocks()
        if "databricks" not in b.identifier
    ]
    definition = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowParameter", "name": "token"}],
        "steps": [
            {
                "type": "roboflow_core/databricks_sql_image_source@v1",
                "name": "db",
                "server_hostname": HOST,
                "http_path": "/sql/1.0/warehouses/abc",
                "access_token": "$inputs.token",
                "catalog": "main",
                "schema_name": "vision",
                "table": "images",
                "metadata_columns": ["id"],
            }
        ],
        "outputs": [
            {"type": "JsonField", "name": "images", "selector": "$steps.db.images"}
        ],
    }
    with pytest.raises(WorkflowDefinitionError) as info:
        parse_workflow_definition(
            raw_workflow_definition=definition, available_blocks=available
        )
    assert "databricks_sql_image_source@v1" in info.value.public_message
    assert "not installed" in info.value.public_message
    assert "step `db`" in info.value.public_message


def test_unified_http_client_is_refused_while_pinned_and_untouched_outside() -> None:
    """Verify unified http client is refused while pinned and untouched outside."""
    from databricks.sql.common.unified_http_client import UnifiedHttpClient

    transport.install_pinned_transport()
    client = UnifiedHttpClient.__new__(UnifiedHttpClient)
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        with pytest.raises(transport.PinnedTransportError, match="refused"):
            with client.request_context(
                mock.MagicMock(value="GET"),
                f"https://{HOST}/api/2.0/connector-service/"
                "feature-flags/PYTHON/x?token=s",
            ):
                pass
    # Outside a pin the original implementation runs (and fails on the bare
    # instance for its own reasons, not ours).
    with pytest.raises(Exception) as info:
        with client.request_context(mock.MagicMock(value="GET"), "https://x/"):
            pass
    assert not isinstance(info.value, transport.PinnedTransportError)


def test_transport_retains_its_pin_when_reopened_on_another_thread() -> None:
    """Verify transport retains its pin when reopened on another thread."""
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        client = _client()
        client.open()

    def _reopen():
        assert transport.current_pin() is None
        client.open()
        client.setCustomHeaders({"User-Agent": "test"})
        result = _pool_of(client)

        return result

    with ThreadPoolExecutor(max_workers=1) as executor:
        pool = executor.submit(_reopen).result(timeout=5)
    assert pool.host == PUBLIC_IP
    assert pool.conn_kw["server_hostname"] == HOST
    assert getattr(client, "_THttpClient__custom_headers")["Host"] == HOST


def test_secondary_guard_does_not_block_an_unowned_thread() -> None:
    """Verify secondary guard does not block an unowned thread."""
    from databricks.sql.common.unified_http_client import UnifiedHttpClient

    transport.install_pinned_transport()
    client = UnifiedHttpClient.__new__(UnifiedHttpClient)

    def _request():
        with pytest.raises(Exception) as info:
            with client.request_context(mock.MagicMock(value="GET"), "https://x/"):
                pass
        assert not isinstance(info.value, transport.PinnedTransportError)

    with transport.pinned_destination(
        transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    ), ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_request).result(timeout=5)


def test_worker_guard_blocks_secondary_requests_after_context_exit(monkeypatch) -> None:
    """Verify worker guard blocks secondary requests after context exit.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
    """
    from databricks.sql.common.unified_http_client import UnifiedHttpClient

    # Undo only the process policy after this test; production workers never reset it.
    monkeypatch.setattr(transport, "_WORKER_ONLY", False)
    transport.install_pinned_transport(worker_only=True)
    client = UnifiedHttpClient.__new__(UnifiedHttpClient)

    def _request():
        assert transport.current_pin() is None
        with pytest.raises(transport.PinnedTransportError, match="refused"):
            with client.request_context(mock.MagicMock(value="GET"), "https://x/"):
                pass

    with ThreadPoolExecutor(max_workers=1) as executor:
        executor.submit(_request).result(timeout=5)


def test_worker_refuses_automatic_kernel_recovery_before_network(monkeypatch) -> None:
    """Verify worker refuses automatic kernel recovery before network.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
    """
    from databricks import sql
    from databricks.sql import client

    monkeypatch.setattr(transport, "_WORKER_ONLY", False)
    transport.install_pinned_transport(worker_only=True)
    monkeypatch.setattr(client, "is_known_reyden", lambda *args: True)
    kernel = mock.MagicMock()
    monkeypatch.setitem(
        sys.modules,
        "databricks.sql.backend.kernel.client",
        mock.Mock(KernelDatabricksClient=kernel),
    )
    with mock.patch.object(socket.socket, "connect") as connect:
        with pytest.raises(transport.PinnedTransportError, match="Thrift backend"):
            sql.connect(
                server_hostname=HOST,
                http_path="/sql/1.0/warehouses/abc",
                access_token="dapi-test-token",
                enable_telemetry=False,
            )
    connect.assert_not_called()
    kernel.assert_not_called()


@pytest.mark.parametrize("backend", ["use_kernel", "use_sea"])
def test_worker_refuses_explicit_non_thrift_backend(monkeypatch, backend) -> None:
    """Verify worker refuses explicit non thrift backend.

    Args:
        monkeypatch (pytest.MonkeyPatch): Scoped replacements restored after this test.
        backend (str): Non-Thrift backend requested to test fail-closed behavior.
    """
    from databricks.sql.session import Session

    monkeypatch.setattr(transport, "_WORKER_ONLY", False)
    transport.install_pinned_transport(worker_only=True)
    with pytest.raises(transport.PinnedTransportError, match="Thrift backend"):
        Session(HOST, "/sql/1.0/warehouses/abc", mock.MagicMock(), **{backend: True})
