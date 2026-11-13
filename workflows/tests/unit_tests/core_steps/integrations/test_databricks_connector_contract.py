"""Contract checks against the real ``databricks-sql-connector``.

The block's other tests fake the connector. These pin the parts of the real
package the block relies on, so a connector release that renames a keyword
argument or a method fails CI instead of production. No network is used.
"""

import inspect

import pytest
from roboflow_workflows.core_steps.integrations.databricks.sql_image_source import (
    transport,
    v1,
)

sql = pytest.importorskip("databricks.sql", reason="databricks extra not installed")


def test_connector_version_is_within_the_declared_range() -> None:
    """Verify connector version is within the declared range."""
    from packaging.version import Version

    version = Version(sql.__version__)
    assert version == Version("4.8.0"), sql.__version__


def test_connect_accepts_every_keyword_the_block_passes() -> None:
    from databricks.sql.backend import thrift_backend
    from databricks.sql.client import Connection

    kwargs = v1.connection_kwargs(
        transport.PinnedDestination(
            hostname="h.cloud.databricks.com", ip_address="52.1.2.3"
        ),
        http_path="/sql/1.0/warehouses/abc",
        timeout_seconds=30,
    )
    connection_source = inspect.getsource(Connection.__init__)
    backend_source = inspect.getsource(thrift_backend)
    for name in kwargs:
        if name in ("server_hostname", "http_path"):
            continue
        spellings = (f'"{name}"', f"{name}:", f"{name}=")
        assert any(
            spelling in connection_source or spelling in backend_source
            for spelling in spellings
        ), f"connector no longer reads `{name}`"


def test_cursor_and_connection_expose_the_methods_the_block_uses() -> None:
    from databricks.sql.client import Connection, Cursor
    from databricks.sql.types import Row

    assert callable(getattr(Cursor, "execute"))
    assert callable(getattr(Cursor, "fetchmany"))
    assert callable(getattr(Cursor, "cancel"))
    assert callable(getattr(Cursor, "__enter__")) and callable(
        getattr(Cursor, "__exit__")
    )
    assert callable(getattr(Connection, "cursor"))
    assert callable(getattr(Connection, "close"))
    assert callable(getattr(Connection, "__enter__")) and callable(
        getattr(Connection, "__exit__")
    )
    assert callable(getattr(Row, "asDict"))
    assert "size" in inspect.signature(Cursor.fetchmany).parameters
    cursor_parameters = inspect.signature(Connection.cursor).parameters
    for name in ("arraysize", "buffer_size_bytes", "row_limit"):
        assert name in cursor_parameters, name


def test_connector_has_no_second_transport_the_pin_does_not_cover() -> None:
    from databricks.sql.common.unified_http_client import UnifiedHttpClient

    assert callable(getattr(UnifiedHttpClient, "request_context"))
    assert callable(getattr(UnifiedHttpClient, "request"))
    source = inspect.getsource(UnifiedHttpClient.request)
    assert "self.request_context(" in source, "request() must route via request_context"


def test_thrift_transport_still_has_the_hooks_the_pin_relies_on() -> None:
    from databricks.sql.auth import thrift_http_client

    base = thrift_http_client.THttpClient
    while issubclass(base, transport._PinnedMixin):
        base = base.__mro__[2] if len(base.__mro__) > 2 else base.__mro__[-1]
    source = inspect.getsource(base)
    assert "def open(self)" in source
    assert "def setCustomHeaders" in source
    assert "__pool = pool_class(self.host, self.port" in source
    assert "def using_proxy" in source
    for attribute in ("host", "port", "scheme", "max_connections", "_ssl_options"):
        assert f"self.{attribute}" in source, attribute
