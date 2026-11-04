"""Pinned HTTPS transport for the Databricks SQL connector.

The block validates ``server_hostname`` (suffix allowlist, DNS answers all
global) before connecting. The connector would then resolve the hostname a
second time, so a DNS answer that changes between validation and connect could
steer the connection to a private address. This module closes that gap at the
connector's own transport: the Thrift HTTP client opens its ``urllib3`` pool on
the validated IP address while keeping the original hostname for TLS SNI,
certificate hostname verification and the HTTP ``Host`` header.

The pin is carried in a ``ContextVar`` that the block sets around ``connect()``.
Outside such a scope the subclass behaves exactly like the connector's class, so
other users of the connector in the same process are unaffected.

Transport inventory for ``databricks-sql-connector`` 4.x (verified against
4.8.0, ``urllib3`` 2.x, ``thrift`` 0.2x):

* ``databricks.sql.auth.thrift_http_client.THttpClient`` - one ``urllib3``
  connection pool per connection, used for every Thrift call: OpenSession,
  ExecuteStatement, GetOperationStatus (polling), GetResultSetMetadata,
  FetchResults, CancelOperation, CloseOperation, CloseSession, including the
  connector's retries and reconnects. Pinned here.
* ``databricks.sql.common.unified_http_client.UnifiedHttpClient`` - a
  ``urllib3`` ``PoolManager`` used for everything else: telemetry export
  (``/telemetry-ext``), the feature-flag lookup
  (``/api/2.0/connector-service/feature-flags/...``), OAuth / token federation,
  CloudFetch result-link downloads and staging (volume ingestion) transfers.
  None of these is needed by the block (PAT auth, telemetry off, cloud fetch
  off, no staging), so while a pin is active this client refuses every request.
* ``databricks.sql.backend.sea`` - the REST (SEA) backend, only with
  ``use_sea=True``; not used.
"""

import contextlib
import contextvars
import ssl
import threading
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional

from urllib3 import HTTPSConnectionPool


@dataclass(frozen=True)
class PinnedDestination:
    """Validated destination the transport must connect to."""

    hostname: str
    ip_address: str
    port: int = 443


_PINNED_DESTINATION: contextvars.ContextVar[Optional[PinnedDestination]] = (
    contextvars.ContextVar("databricks_pinned_destination", default=None)
)
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
# Pins are scoped to the calling context, but the connector hands work to its
# own threads (CloudFetch downloads, feature-flag refresh, telemetry flush)
# where a ContextVar is invisible. This process-wide count of active pins lets
# the guards refuse those paths from any thread while a block run is active.
_ACTIVE_PINS = 0
_ACTIVE_PINS_LOCK = threading.Lock()


class PinnedTransportError(RuntimeError):
    """Raised when the transport would connect somewhere other than the pin."""


@contextlib.contextmanager
def pinned_destination(destination: PinnedDestination) -> Iterator[None]:
    """Scope within which the connector's transport connects to ``destination``.

    Args:
        destination: Hostname, validated IP address and port to connect to.

    Yields:
        Nothing; the pin is active for the calling context until exit.
    """
    global _ACTIVE_PINS
    token = _PINNED_DESTINATION.set(destination)
    with _ACTIVE_PINS_LOCK:
        _ACTIVE_PINS += 1
    try:
        yield
    finally:
        with _ACTIVE_PINS_LOCK:
            _ACTIVE_PINS -= 1
        _PINNED_DESTINATION.reset(token)


def pin_active_anywhere() -> bool:
    """Return True while any context in the process holds a pin."""
    with _ACTIVE_PINS_LOCK:
        return _ACTIVE_PINS > 0


def current_pin() -> Optional[PinnedDestination]:
    """Return the destination pinned for the calling context, if any."""
    return _PINNED_DESTINATION.get()


def install_pinned_transport() -> type:
    """Replace the connector's Thrift HTTP client with the pinning subclass.

    Idempotent and process-wide. The subclass only changes behaviour while a
    :func:`pinned_destination` scope is active in the calling context.

    Returns:
        The transport class the connector now instantiates.

    Raises:
        RuntimeError: If ``databricks-sql-connector`` is not installed.
    """
    global _INSTALLED
    with _INSTALL_LOCK:
        try:
            # The connector's auth module references `thrift.transport` as an
            # attribute of the `thrift` package, which only exists once the
            # submodule has been imported somewhere; do it explicitly.
            import thrift.transport.THttpClient  # noqa: F401
            from databricks.sql.auth import thrift_http_client
        except (
            ImportError
        ) as error:  # pragma: no cover - exercised by availability tests
            raise RuntimeError("databricks-sql-connector is not installed") from error

        if (
            _INSTALLED
            and isinstance(thrift_http_client.THttpClient, type)
            and issubclass(thrift_http_client.THttpClient, _PinnedMixin)
        ):
            return thrift_http_client.THttpClient

        base = thrift_http_client.THttpClient
        if issubclass(base, _PinnedMixin):
            _guard_unified_http_client()
            _guard_result_link_downloads()
            _INSTALLED = True
            return base

        class PinnedTHttpClient(_PinnedMixin, base):  # type: ignore[misc,valid-type]
            pass

        PinnedTHttpClient.__name__ = base.__name__
        PinnedTHttpClient.__qualname__ = base.__qualname__
        thrift_http_client.THttpClient = PinnedTHttpClient
        _guard_unified_http_client()
        _guard_result_link_downloads()
        _INSTALLED = True
        return PinnedTHttpClient


def _guard_unified_http_client() -> None:
    # The connector's second HTTP client never carries a request the block
    # needs (see the module docstring). Refusing it while pinned means no
    # telemetry, feature-flag, OAuth, cloud-fetch or staging request can leave
    # the process on the block's behalf, whatever the warehouse answers.
    from databricks.sql.common import unified_http_client

    client_class = unified_http_client.UnifiedHttpClient
    if getattr(client_class, "_roboflow_pinned_guard", False):
        return
    original_request_context = client_class.request_context

    def request_context(self, method, url, headers=None, **kwargs):
        if _PINNED_DESTINATION.get() is not None or pin_active_anywhere():
            raise PinnedTransportError(
                f"request to {_redact_url(url)} refused: the Databricks block only "
                "speaks to the pinned warehouse over the Thrift transport"
            )
        return original_request_context(self, method, url, headers=headers, **kwargs)

    client_class.request_context = request_context
    client_class._roboflow_pinned_guard = True


def _guard_result_link_downloads() -> None:
    # `use_cloud_fetch=False` asks the warehouse for inline results, but the
    # response format is the warehouse's choice: a result set that arrives as
    # URL_BASED_SET would otherwise start download threads for whatever links
    # it carries. Refuse the queue itself, before any thread is spawned, so the
    # policy does not rest on PyArrow being absent.
    from databricks.sql import utils as connector_utils

    queue_class = connector_utils.ThriftCloudFetchQueue
    if getattr(queue_class, "_roboflow_pinned_guard", False):
        return

    class PinnedThriftCloudFetchQueue(queue_class):  # type: ignore[misc,valid-type]
        _roboflow_pinned_guard = True

        def __init__(self, *args, **kwargs):
            if _PINNED_DESTINATION.get() is not None or pin_active_anywhere():
                raise PinnedTransportError(
                    "the warehouse returned result links (cloud fetch) although "
                    "inline results were requested; downloading them is refused"
                )
            super().__init__(*args, **kwargs)

    PinnedThriftCloudFetchQueue.__name__ = queue_class.__name__
    PinnedThriftCloudFetchQueue.__qualname__ = queue_class.__qualname__
    connector_utils.ThriftCloudFetchQueue = PinnedThriftCloudFetchQueue


def _redact_url(url: Any) -> str:
    text = str(url)
    return text.split("?", 1)[0][:200]


class _PinnedMixin:
    """Behaviour mixed into the connector's ``THttpClient``.

    ``open()`` is the single place the connector creates its connection pool;
    every request, retry and same-host redirect goes through that pool, so
    pinning it pins them all. A redirect to another host is refused by
    ``urllib3`` (``assert_same_host``), and the block additionally sets the
    connector's redirect budget to zero.
    """

    def open(self):  # noqa: D102 - connector interface
        pin = _PINNED_DESTINATION.get()
        if pin is None:
            return super().open()  # type: ignore[misc]
        if self.scheme != "https":
            raise PinnedTransportError(
                "pinned transport requires https, got " + str(self.scheme)
            )
        if self.using_proxy():
            raise PinnedTransportError(
                "pinned transport refuses to use an HTTP(S) proxy: the proxy would "
                "resolve the destination outside the validating block"
            )
        expected_port = pin.port
        actual_port = self.port or 443
        if self.host != pin.hostname or actual_port != expected_port:
            raise PinnedTransportError(
                f"transport target {self.host}:{actual_port} does not match the "
                f"validated destination {pin.hostname}:{expected_port}"
            )
        ssl_options = self._ssl_options
        pool_kwargs: Dict[str, Any] = {
            "maxsize": self.max_connections,
            "cert_reqs": ssl.CERT_REQUIRED,
            "server_hostname": pin.hostname,
            "assert_hostname": pin.hostname,
        }
        if ssl_options is not None:
            if not ssl_options.tls_verify or not getattr(
                ssl_options, "tls_verify_hostname", True
            ):
                raise PinnedTransportError(
                    "pinned transport requires TLS certificate and hostname verification"
                )
            pool_kwargs.update(
                {
                    "ca_certs": ssl_options.tls_trusted_ca_file,
                    "cert_file": ssl_options.tls_client_cert_file,
                    "key_file": ssl_options.tls_client_cert_key_file,
                    "key_password": ssl_options.tls_client_cert_key_password,
                }
            )
        pool = HTTPSConnectionPool(pin.ip_address, expected_port, **pool_kwargs)
        self._set_pool(pool)
        return None

    def setCustomHeaders(
        self, headers: Dict[str, str]
    ):  # noqa: N802 - connector interface
        pin = _PINNED_DESTINATION.get()
        if pin is not None:
            headers = dict(headers)
            headers["Host"] = pin.hostname
        return super().setCustomHeaders(headers)  # type: ignore[misc]

    def _set_pool(self, pool: HTTPSConnectionPool) -> None:
        # The connector stores the pool in a name-mangled attribute of its own
        # class (`_THttpClient__pool`); write the same slot it reads.
        base_name = None
        for klass in type(self).__mro__:
            if klass.__name__ == "THttpClient" and not issubclass(klass, _PinnedMixin):
                base_name = klass.__name__
                break
        attribute = f"_{base_name or 'THttpClient'}__pool"
        setattr(self, attribute, pool)
