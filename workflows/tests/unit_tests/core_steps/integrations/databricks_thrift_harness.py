"""A local HTTPS warehouse that speaks enough Thrift for the real connector.

The real-transport tests run the actual ``databricks-sql-connector`` against
this server through the block's pinned transport, so what they observe is the
connector's real network behaviour: which addresses it connects to, which SNI
and ``Host`` it presents, how it reacts to retries, reconnects, redirects,
result links and bad certificates, and whether cancellation reaches the server.

The server records every TCP connection (peer, SNI), every HTTP request
(path, headers, Thrift method) and the operation state, and can be scripted
per request: an HTTP status to return instead of a Thrift reply, a redirect, a
dropped connection, a hang, result links instead of inline rows, or holding
the statement in the RUNNING state until it is cancelled.

TLS uses a freshly generated, short-lived self-signed certificate in a private
temporary directory. The harness certificate is trusted only by the test client.
"""

import http.server
import socketserver
import ssl
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from databricks.sql.thrift_api.TCLIService import TCLIService, ttypes
from thrift.protocol import TBinaryProtocol
from thrift.transport import TTransport

HOST = "dbc-a1b2c3d4-e5f6.cloud.databricks.com"
OTHER_HOST = "other.cloud.databricks.com"
HTTP_PATH = "/sql/1.0/warehouses/abcdef1234567890"
COLUMN_TYPES = {
    "string": ttypes.TTypeId.STRING_TYPE,
    "binary": ttypes.TTypeId.BINARY_TYPE,
    "bigint": ttypes.TTypeId.BIGINT_TYPE,
}


@dataclass
class HttpAction:
    """What to do with one HTTP request instead of answering the Thrift call."""

    status: Optional[int] = None
    headers: Dict[str, str] = field(default_factory=dict)
    close_connection: bool = False
    hang_seconds: float = 0.0


@dataclass
class Behaviour:
    """Scripted behaviour of the fake warehouse."""

    columns: Sequence[Tuple[str, str]] = (("id", "bigint"), ("bytes", "binary"))
    rows: List[Dict[str, Any]] = field(default_factory=list)
    chunk_rows: int = 8
    # Per-request HTTP actions consumed in order; ``None`` means a normal reply.
    http_script: List[Optional[HttpAction]] = field(default_factory=list)
    # Keep the statement RUNNING until CancelOperation arrives.
    hold_running: bool = False
    # Return result links (URL_BASED_SET) instead of inline rows.
    result_links: List[str] = field(default_factory=list)


@dataclass
class Observations:
    connections: List[Dict[str, Any]] = field(default_factory=list)
    requests: List[Dict[str, Any]] = field(default_factory=list)
    operation_states: List[str] = field(default_factory=list)
    cancelled: threading.Event = field(default_factory=threading.Event)
    session_closed: threading.Event = field(default_factory=threading.Event)

    @property
    def thrift_methods(self) -> List[str]:
        return [r["method"] for r in self.requests if r.get("method")]


def _status() -> ttypes.TStatus:
    return ttypes.TStatus(statusCode=ttypes.TStatusCode.SUCCESS_STATUS)


def _handle_identifier() -> ttypes.THandleIdentifier:
    return ttypes.THandleIdentifier(guid=uuid.uuid4().bytes, secret=uuid.uuid4().bytes)


class _WarehouseHandler(TCLIService.Iface):
    def __init__(self, behaviour: Behaviour, observations: Observations):
        self.behaviour = behaviour
        self.observations = observations
        self._lock = threading.Lock()
        self._cursor = 0
        self._operation = _handle_identifier()
        self._cancelled = False

    def OpenSession(self, req):
        return ttypes.TOpenSessionResp(
            status=_status(),
            serverProtocolVersion=ttypes.TProtocolVersion.SPARK_CLI_SERVICE_PROTOCOL_V7,
            sessionHandle=ttypes.TSessionHandle(sessionId=_handle_identifier()),
        )

    def CloseSession(self, req):
        self.observations.session_closed.set()
        return ttypes.TCloseSessionResp(status=_status())

    def ExecuteStatement(self, req):
        with self._lock:
            self._cursor = 0
            self._cancelled = False
            self._operation = _handle_identifier()
            self.observations.operation_states.append("RUNNING")
        return ttypes.TExecuteStatementResp(
            status=_status(),
            operationHandle=ttypes.TOperationHandle(
                operationId=self._operation,
                operationType=ttypes.TOperationType.EXECUTE_STATEMENT,
                hasResultSet=True,
            ),
            directResults=None,
        )

    def GetOperationStatus(self, req):
        if self._cancelled:
            state = ttypes.TOperationState.CANCELED_STATE
        elif self.behaviour.hold_running:
            time.sleep(0.05)
            state = ttypes.TOperationState.RUNNING_STATE
        else:
            state = ttypes.TOperationState.FINISHED_STATE
        return ttypes.TGetOperationStatusResp(status=_status(), operationState=state)

    def CancelOperation(self, req):
        with self._lock:
            self._cancelled = True
            self.observations.operation_states.append("CANCELED")
        self.observations.cancelled.set()
        return ttypes.TCancelOperationResp(status=_status())

    def CloseOperation(self, req):
        self.observations.operation_states.append("CLOSED")
        return ttypes.TCloseOperationResp(status=_status())

    def GetResultSetMetadata(self, req):
        return self._metadata()

    def FetchResults(self, req):
        if self.behaviour.result_links:
            links = [
                ttypes.TSparkArrowResultLink(
                    fileLink=link,
                    expiryTime=int(time.time()) + 3600,
                    startRowOffset=0,
                    rowCount=1,
                    bytesNum=1,
                )
                for link in self.behaviour.result_links
            ]
            return ttypes.TFetchResultsResp(
                status=_status(),
                hasMoreRows=False,
                results=ttypes.TRowSet(startRowOffset=0, rows=[], resultLinks=links),
                resultSetMetadata=self._metadata(ttypes.TSparkRowSetType.URL_BASED_SET),
            )
        with self._lock:
            start = self._cursor
            chunk = self.behaviour.rows[start : start + self.behaviour.chunk_rows]
            self._cursor = start + len(chunk)
            has_more = self._cursor < len(self.behaviour.rows)
        columns = [
            self._column(name, kind, chunk) for name, kind in self.behaviour.columns
        ]
        return ttypes.TFetchResultsResp(
            status=_status(),
            hasMoreRows=has_more,
            results=ttypes.TRowSet(startRowOffset=start, rows=[], columns=columns),
            resultSetMetadata=self._metadata(),
        )

    def _metadata(self, result_format=ttypes.TSparkRowSetType.COLUMN_BASED_SET):
        schema = ttypes.TTableSchema(
            columns=[
                ttypes.TColumnDesc(
                    columnName=name,
                    typeDesc=ttypes.TTypeDesc(
                        types=[
                            ttypes.TTypeEntry(
                                primitiveEntry=ttypes.TPrimitiveTypeEntry(
                                    type=COLUMN_TYPES[kind]
                                )
                            )
                        ]
                    ),
                    position=position,
                )
                for position, (name, kind) in enumerate(self.behaviour.columns, 1)
            ]
        )
        return ttypes.TGetResultSetMetadataResp(
            status=_status(),
            schema=schema,
            resultFormat=result_format,
            lz4Compressed=False,
            isStagingOperation=False,
        )

    @staticmethod
    def _column(name: str, kind: str, rows: List[Dict[str, Any]]) -> ttypes.TColumn:
        values = [row[name] for row in rows]
        nulls = b"\x00"
        if kind == "binary":
            return ttypes.TColumn(
                binaryVal=ttypes.TBinaryColumn(values=values, nulls=nulls)
            )
        if kind == "bigint":
            return ttypes.TColumn(i64Val=ttypes.TI64Column(values=values, nulls=nulls))
        return ttypes.TColumn(
            stringVal=ttypes.TStringColumn(values=[str(v) for v in values], nulls=nulls)
        )


class _RequestHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: "FakeWarehouse"

    def log_message(self, *args: Any) -> None:  # noqa: D102 - quiet
        return

    def do_POST(self) -> None:  # noqa: N802 - http.server interface
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        record = {
            "path": self.path,
            "host_header": self.headers.get("Host"),
            "headers": dict(self.headers),
            "method": self._thrift_method(body),
        }
        self.server.observations.requests.append(record)
        action = self.server.next_action()
        if action is not None:
            if action.hang_seconds:
                time.sleep(action.hang_seconds)
            if action.close_connection:
                self.close_connection = True
                self.connection.close()
                return
            self.send_response(action.status or 500)
            for key, value in action.headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        output = TTransport.TMemoryBuffer()
        processor = TCLIService.Processor(self.server.handler)
        processor.process(
            TBinaryProtocol.TBinaryProtocol(TTransport.TMemoryBuffer(body)),
            TBinaryProtocol.TBinaryProtocol(output),
        )
        payload = output.getvalue()
        self.send_response(200)
        self.send_header("Content-Type", "application/x-thrift")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _thrift_method(body: bytes) -> Optional[str]:
        try:
            protocol = TBinaryProtocol.TBinaryProtocol(TTransport.TMemoryBuffer(body))
            name, _, _ = protocol.readMessageBegin()
            return name
        except Exception:
            return None


class FakeWarehouse(socketserver.ThreadingMixIn, http.server.HTTPServer):
    """HTTPS Thrift server bound to 127.0.0.1 on a free port."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, behaviour: Behaviour, certificate_host: str = HOST):
        self._tls_directory = tempfile.TemporaryDirectory(prefix="databricks-test-tls-")
        tls_dir = Path(self._tls_directory.name)
        configuration = tls_dir / "openssl.cnf"
        configuration.write_text(
            "[req]\nprompt=no\ndistinguished_name=subject\nx509_extensions=extensions\n"
            f"[subject]\nCN={certificate_host}\n"
            "[extensions]\nbasicConstraints=critical,CA:TRUE\n"
            "keyUsage=digitalSignature,keyEncipherment,keyCertSign\n"
            "extendedKeyUsage=serverAuth\n"
            f"subjectAltName=DNS:{certificate_host}\n"
        )
        self.ca_file = str(tls_dir / "certificate.pem")
        key_file = tls_dir / "key.pem"
        try:
            subprocess.run(
                [
                    "openssl",
                    "req",
                    "-x509",
                    "-newkey",
                    "rsa:2048",
                    "-nodes",
                    "-days",
                    "2",
                    "-config",
                    str(configuration),
                    "-keyout",
                    str(key_file),
                    "-out",
                    self.ca_file,
                ],
                check=True,
                capture_output=True,
                timeout=10,
            )
        except BaseException:
            self._tls_directory.cleanup()
            raise

        super().__init__(("127.0.0.1", 0), _RequestHandler)
        self.behaviour = behaviour
        self.observations = Observations()
        self.handler = _WarehouseHandler(behaviour, self.observations)
        self._script_lock = threading.Lock()
        self._script_index = 0
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=self.ca_file, keyfile=str(key_file))
        observations = self.observations

        def sni_callback(sock, server_name, _context):
            observations.connections.append(
                {"sni": server_name, "peer": sock.getpeername()[0]}
            )

        context.sni_callback = sni_callback
        self.socket = context.wrap_socket(self.socket, server_side=True)
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self.server_address[1]

    def next_action(self) -> Optional[HttpAction]:
        with self._script_lock:
            script = self.behaviour.http_script
            if self._script_index < len(script):
                action = script[self._script_index]
                self._script_index += 1
                return action
            return None

    def __enter__(self) -> "FakeWarehouse":
        self._thread.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join(timeout=5)
        self._tls_directory.cleanup()

    def handle_error(self, request, client_address) -> None:
        # TLS handshake failures (bad certificate tests) are expected noise.
        return
