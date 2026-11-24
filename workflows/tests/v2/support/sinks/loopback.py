"""In-process fakes and a loopback HTTP server for sink tests.

Nothing here reaches beyond this process or ``127.0.0.1``::

    RecordingWebhookClient   records WebhookRequest objects; can fail or wait
    LoopbackHttpServer       real HTTP on 127.0.0.1:<free port>; records requests
    ManualClock              seconds that advance only when a test says so
"""

import json
import threading
import urllib.parse
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional


class RecordingWebhookClient:
    """Fake ``webhook_client``: records requests instead of sending them.

    Args:
        error: Raised by every call after recording, when set.
        gate: Every call waits for this event first, when set.
    """

    def __init__(
        self,
        *,
        error: Optional[Exception] = None,
        gate: Optional[threading.Event] = None,
    ) -> None:
        self.requests: List[Any] = []
        self._error = error
        self._gate = gate
        self._lock = threading.Lock()

    def __call__(self, request: Any) -> None:
        if self._gate is not None:
            self._gate.wait(timeout=10)
        with self._lock:
            self.requests.append(request)
        if self._error is not None:
            raise self._error


class ManualClock:
    """Monotonic-style clock for cooldown and interval tests.

    Args:
        start: Initial reading in seconds.
    """

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """Move the clock forward by ``seconds``."""
        self.now += seconds


@dataclass(frozen=True)
class ReceivedRequest:
    """One request seen by ``LoopbackHttpServer``.

    Args:
        method: HTTP method.
        path: Path without query.
        query: Parsed query values.
        headers: Header values, lower-case names.
        body: Raw body bytes.
    """

    method: str
    path: str
    query: Dict[str, List[str]]
    headers: Dict[str, str]
    body: bytes

    def json(self) -> Any:
        """Return the body parsed as JSON."""
        parsed = json.loads(self.body)

        return parsed


class LoopbackHttpServer:
    """HTTP server on ``127.0.0.1`` with a free port, for real-transport tests.

    Use as a context manager. Every request gets ``status``; a 3xx also gets
    a ``Location`` header pointing back to the server.

    Args:
        status: Response status code.
    """

    def __init__(self, *, status: int = 200) -> None:
        self.status = status
        self.received: List[ReceivedRequest] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def _handle(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                parts = urllib.parse.urlsplit(self.path)
                server.received.append(
                    ReceivedRequest(
                        method=self.command,
                        path=parts.path,
                        query=urllib.parse.parse_qs(parts.query),
                        headers={k.lower(): v for k, v in self.headers.items()},
                        body=self.rfile.read(length),
                    )
                )
                self.send_response(server.status)
                if 300 <= server.status < 400:
                    self.send_header("Location", server.url("/elsewhere"))
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = do_PUT = _handle

            def log_message(self, *args: Any) -> None:
                return None

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    def url(self, path: str = "/hook") -> str:
        """Return the server URL of ``path``."""
        host, port = self._httpd.server_address[:2]
        address = f"http://{host}:{port}{path}"

        return address

    def __enter__(self) -> "LoopbackHttpServer":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)
