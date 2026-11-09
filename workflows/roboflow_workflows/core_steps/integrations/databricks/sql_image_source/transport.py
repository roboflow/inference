"""Pinned HTTPS transport for the Databricks SQL connector.

The block validates ``server_hostname`` (suffix allowlist, DNS answers all
global) before connecting. The connector would then resolve the hostname a
second time, so a DNS answer that changes between validation and connect could
steer the connection to a private address. This module closes that gap at the
connector's own transport: the Thrift HTTP client opens its ``urllib3`` pool on
the validated IP address while keeping the original hostname for TLS SNI,
certificate hostname verification and the HTTP ``Host`` header.

The dedicated worker installs these hooks before connecting. A transport captures
its pin at construction and retains it across reconnects and cancellation threads.
The worker refuses secondary HTTP paths for its entire lifetime. The parent
process never installs these hooks.

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
  off, no staging), so the dedicated worker refuses every such request.
* REST (SEA) and native kernel backends, including automatic kernel recovery,
  are refused before session construction; neither uses the pinned Thrift pool.
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
    max_response_bytes: int = 64 * 1024 * 1024 + 65536


_PINNED_DESTINATION: contextvars.ContextVar[Optional[PinnedDestination]] = (
    contextvars.ContextVar("databricks_pinned_destination", default=None)
)
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False
_WORKER_ONLY = False
# Diagnostic count for checking context cleanup. Network policy uses the
# dedicated worker lifetime, so background threads cannot outlive protection.
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


def install_pinned_transport(*, worker_only: bool = False) -> type:
    """Replace the connector's Thrift HTTP client with the pinning subclass.

    Idempotent and process-wide; production callers must use a dedicated worker.

    Args:
        worker_only: Permanently refuse secondary requests in this process, even
            from background threads and after the construction context exits.

    Returns:
        The transport class the connector now instantiates.

    Raises:
        RuntimeError: If ``databricks-sql-connector`` is not installed.
    """
    global _INSTALLED, _WORKER_ONLY
    with _INSTALL_LOCK:
        _WORKER_ONLY = _WORKER_ONLY or worker_only
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

        _guard_response_parsing()
        if worker_only:
            from databricks.sql.backend import thrift_backend

            # Request column-based Thrift results even if the host installed Arrow.
            # Only the disposable interpreter changes this connector global.
            thrift_backend.pyarrow = None

        if (
            _INSTALLED
            and isinstance(thrift_http_client.THttpClient, type)
            and issubclass(thrift_http_client.THttpClient, _PinnedMixin)
        ):
            return thrift_http_client.THttpClient

        base = thrift_http_client.THttpClient
        if issubclass(base, _PinnedMixin):
            _guard_non_thrift_backends()
            _guard_unified_http_client()
            _guard_result_link_downloads()
            _INSTALLED = True
            return base

        class PinnedTHttpClient(_PinnedMixin, base):  # type: ignore[misc,valid-type]
            pass

        PinnedTHttpClient.__name__ = base.__name__
        PinnedTHttpClient.__qualname__ = base.__qualname__
        thrift_http_client.THttpClient = PinnedTHttpClient
        _guard_non_thrift_backends()
        _guard_unified_http_client()
        _guard_result_link_downloads()
        _INSTALLED = True
        return PinnedTHttpClient


def _guard_non_thrift_backends() -> None:
    # Connector 4.8 can automatically retry a rejected Thrift session using its
    # native kernel transport. Reject that switch before any backend is created.
    from databricks.sql.session import Session

    if getattr(Session, "_roboflow_backend_guard", False):
        return
    original_init = Session.__init__

    def init(self, *args, **kwargs):
        if (_WORKER_ONLY or _PINNED_DESTINATION.get() is not None) and (
            kwargs.get("use_kernel") or kwargs.get("use_sea")
        ):
            raise PinnedTransportError(
                "the Databricks worker requires the Thrift backend"
            )
        original_init(self, *args, **kwargs)

    Session.__init__ = init
    Session._roboflow_backend_guard = True


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
        if _WORKER_ONLY or _PINNED_DESTINATION.get() is not None:
            raise PinnedTransportError(
                "secondary request refused: the Databricks block only "
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
            if _WORKER_ONLY or _PINNED_DESTINATION.get() is not None:
                raise PinnedTransportError(
                    "the warehouse returned result links (cloud fetch) although "
                    "inline results were requested; downloading them is refused"
                )
            super().__init__(*args, **kwargs)

    PinnedThriftCloudFetchQueue.__name__ = queue_class.__name__
    PinnedThriftCloudFetchQueue.__qualname__ = queue_class.__qualname__
    connector_utils.ThriftCloudFetchQueue = PinnedThriftCloudFetchQueue


def _guard_response_parsing() -> None:
    from databricks.sql.thrift_api.TCLIService import ttypes
    from thrift.protocol import TBinaryProtocol

    original = TBinaryProtocol.TBinaryProtocol
    if getattr(original, "_roboflow_bounded", False):
        return

    class BoundedProtocol(original):
        _roboflow_bounded = True

        def __init__(self, trans, *args, **kwargs):
            pin = getattr(trans, "_roboflow_destination", None)
            self._bounded = pin is not None
            self._depth = 0
            self._items = 0
            if pin is not None:
                kwargs["string_length_limit"] = pin.max_response_bytes
                kwargs["container_length_limit"] = 65536
            super().__init__(trans, *args, **kwargs)

        def readMessageBegin(self):
            self._depth = 0
            self._items = 0
            return super().readMessageBegin()

        def _enter(self, items=1):
            if self._bounded:
                self._depth += 1
                self._items += items
                if self._depth > 64 or self._items > 131072:
                    raise PinnedTransportError(
                        "warehouse response structure exceeds its limit"
                    )

        def _leave(self):
            if self._bounded:
                self._depth -= 1

        def readStructBegin(self):
            self._enter()
            return super().readStructBegin()

        def readStructEnd(self):
            self._leave()
            return super().readStructEnd()

        def readListBegin(self):
            result = super().readListBegin()
            self._enter(result[1])
            return result

        def readListEnd(self):
            self._leave()
            return super().readListEnd()

        def readSetBegin(self):
            result = super().readSetBegin()
            self._enter(result[1])
            return result

        def readSetEnd(self):
            self._leave()
            return super().readSetEnd()

        def readMapBegin(self):
            result = super().readMapBegin()
            self._enter(2 * result[2])
            return result

        def readMapEnd(self):
            self._leave()
            return super().readMapEnd()

    TBinaryProtocol.TBinaryProtocol = BoundedProtocol
    original_metadata_read = ttypes.TGetResultSetMetadataResp.read

    def read_metadata(self, protocol):
        original_metadata_read(self, protocol)
        if getattr(protocol, "_bounded", False) and (
            self.arrowSchema
            or self.lz4Compressed
            or self.resultFormat not in (None, ttypes.TSparkRowSetType.COLUMN_BASED_SET)
        ):
            raise PinnedTransportError("only column-based Thrift results are allowed")

    ttypes.TGetResultSetMetadataResp.read = read_metadata
    original_row_read = ttypes.TRowSet.read

    def read_rows(self, protocol):
        original_row_read(self, protocol)
        if getattr(protocol, "_bounded", False) and (
            self.arrowBatches or self.resultLinks
        ):
            raise PinnedTransportError(
                "Arrow, LZ4 and result links (cloud fetch) are refused"
            )

    ttypes.TRowSet.read = read_rows


class _PinnedMixin:
    """Behaviour mixed into the connector's ``THttpClient``.

    The pin belongs to this client, including when another thread reopens it.
    Every request uses a pool which refuses all redirects independently of the
    connector's retry policy.
    """

    def __init__(self, *args, **kwargs):
        self._roboflow_destination = _PINNED_DESTINATION.get()
        if _WORKER_ONLY and self._roboflow_destination is None:
            raise PinnedTransportError(
                "worker transport requires a validated destination"
            )
        super().__init__(*args, **kwargs)

    def open(self):  # noqa: D102 - connector interface
        pin = self._roboflow_destination
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
                    "pinned transport requires TLS certificate "
                    "and hostname verification"
                )
            pool_kwargs.update(
                {
                    "ca_certs": ssl_options.tls_trusted_ca_file,
                    "cert_file": ssl_options.tls_client_cert_file,
                    "key_file": ssl_options.tls_client_cert_key_file,
                    "key_password": ssl_options.tls_client_cert_key_password,
                }
            )
        pool = _PinnedHTTPSConnectionPool(pin.ip_address, expected_port, **pool_kwargs)
        self._set_pool(pool)
        return None

    def setCustomHeaders(
        self, headers: Dict[str, str]
    ):  # noqa: N802 - connector interface
        pin = self._roboflow_destination
        if pin is not None:
            headers = dict(headers)
            headers["Host"] = pin.hostname
            headers["Accept-Encoding"] = "identity"
        return super().setCustomHeaders(headers)  # type: ignore[misc]

    def flush(self):
        self._roboflow_read_bytes = 0
        super().flush()
        pin = self._roboflow_destination
        if pin is None:
            return
        response = self._THttpClient__resp
        encoding = response.headers.get("Content-Encoding", "identity").lower()
        length = response.headers.get("Content-Length")
        try:
            if encoding != "identity":
                raise PinnedTransportError(
                    "compressed warehouse HTTP responses are refused"
                )
            if length is not None and (
                int(length) < 0 or int(length) > pin.max_response_bytes
            ):
                raise PinnedTransportError("warehouse response exceeds its byte limit")
        except (ValueError, PinnedTransportError):
            self.close()
            raise

    def read(self, size):
        pin = self._roboflow_destination
        if pin is None:
            return super().read(size)
        remaining = pin.max_response_bytes - self._roboflow_read_bytes
        amount = min(size, remaining + 1) if size >= 0 else remaining + 1
        data = self._THttpClient__resp.read(amount, decode_content=False)
        self._roboflow_read_bytes += len(data)
        if len(data) > remaining:
            self.close()
            raise PinnedTransportError("warehouse response exceeds its byte limit")
        return data

    def close(self):
        if self._roboflow_destination is None:
            return super().close()
        response = getattr(self, "_THttpClient__resp", None)
        if response is not None:
            # Never drain an attacker-controlled body during error cleanup.
            response.close()
            response.release_conn()
            self._THttpClient__resp = None

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


class _PinnedHTTPSConnectionPool(HTTPSConnectionPool):
    """Disable redirects at urllib3's request boundary, including relative URLs."""

    def urlopen(self, method, url, *args, **kwargs):
        # urllib3's reconnect path passes body, headers, retries and redirect
        # positionally; avoid supplying redirect twice on that recursive call.
        if len(args) >= 4:
            args = (*args[:3], False, *args[4:])
            kwargs.pop("redirect", None)
        else:
            kwargs["redirect"] = False
        response = super().urlopen(method, url, *args, **kwargs)
        if 300 <= response.status < 400:
            # Do not drain an untrusted response body while rejecting a redirect.
            response.close()
            response.release_conn()
            raise PinnedTransportError("warehouse redirects are refused")

        return response
