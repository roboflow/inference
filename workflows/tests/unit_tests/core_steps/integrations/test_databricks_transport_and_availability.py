"""Transport pinning, availability gating and explicit rejection for Databricks.

The pinning tests use the real ``databricks-sql-connector`` transport class
when the package is installed (it is in the workflows test extras); they open
no network connection, they inspect the pool the transport would use.
"""

import socket
import sys
from typing import Any
from unittest import mock

import pytest
from roboflow_workflows.core_steps.integrations.databricks import availability
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    transport,
)
from roboflow_workflows.errors import WorkflowDefinitionError
from roboflow_workflows.execution_engine.v1.compiler.syntactic_parser import (
    parse_workflow_definition,
)

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


def test_pinned_pool_connects_to_the_validated_ip_with_the_hostname_for_tls_and_host() -> (
    None
):
    client = _client()
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
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
    client = _client()
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with mock.patch.object(
        socket,
        "getaddrinfo",
        return_value=[(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PRIVATE_IP, 443))],
    ) as resolver, transport.pinned_destination(pin):
        client.open()
        pool = _pool_of(client)
    assert (
        pool.host == PUBLIC_IP
    ), "pool must use the validated address, not a fresh lookup"
    resolver.assert_not_called()


def test_transport_refuses_a_target_that_differs_from_the_pin() -> None:
    client = _client("other.cloud.databricks.com")
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        with pytest.raises(transport.PinnedTransportError, match="does not match"):
            client.open()


def test_transport_refuses_proxies_and_disabled_tls_verification_while_pinned() -> None:
    from databricks.sql.auth import thrift_http_client
    from databricks.sql.types import SSLOptions

    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    client = _client()
    with transport.pinned_destination(pin), mock.patch.object(
        client, "using_proxy", return_value=True
    ):
        with pytest.raises(transport.PinnedTransportError, match="proxy"):
            client.open()
    insecure = thrift_http_client.THttpClient(
        auth_provider=mock.MagicMock(),
        uri_or_host=f"https://{HOST}:443/sql/1.0/warehouses/abc",
        ssl_options=SSLOptions(tls_verify=False),
    )
    with transport.pinned_destination(pin):
        with pytest.raises(transport.PinnedTransportError, match="verification"):
            insecure.open()


def test_cross_host_redirects_are_refused_by_the_pinned_pool() -> None:
    client = _client()
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
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
) -> None:
    from roboflow_workflows.core_steps import loader
    from roboflow_workflows.execution_engine.introspection import blocks_loader

    monkeypatch.setattr(
        loader,
        "databricks_unavailability_reason",
        lambda: "the `databricks-sql-connector` package is not installed",
    )
    (
        blocks_loader.load_core_workflow_blocks.cache_clear()
        if hasattr(blocks_loader.load_core_workflow_blocks, "cache_clear")
        else None
    )
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
                "sql_statement": "SELECT 1",
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
    from databricks.sql.common.unified_http_client import UnifiedHttpClient

    transport.install_pinned_transport()
    client = UnifiedHttpClient.__new__(UnifiedHttpClient)
    pin = transport.PinnedDestination(hostname=HOST, ip_address=PUBLIC_IP)
    with transport.pinned_destination(pin):
        with pytest.raises(transport.PinnedTransportError, match="refused"):
            with client.request_context(
                mock.MagicMock(value="GET"),
                f"https://{HOST}/api/2.0/connector-service/feature-flags/PYTHON/x?token=s",
            ):
                pass
    # Outside a pin the original implementation runs (and fails on the bare
    # instance for its own reasons, not ours).
    with pytest.raises(Exception) as info:
        with client.request_context(mock.MagicMock(value="GET"), "https://x/"):
            pass
    assert not isinstance(info.value, transport.PinnedTransportError)
