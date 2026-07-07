"""Local stand-in for the Modal ``webexec`` app used by integration tests.

Serves the exact same execution code as production: it imports
``modal/webexec_runtime.py`` (the module ``modal/modal_app.py`` ships into the
Modal image) and only adds the transport glue Modal normally provides — the
HTTP POST route, the ``/ws`` websocket app, and a ``/health`` probe. Tests
point ``MODAL_WEB_ENDPOINT_URL`` / ``MODAL_WS_ENDPOINT_URL`` at this server
(see the ``local_webexec_server`` fixture in ``conftest.py``).

Run with:
    uvicorn local_webexec_app:app --port <port>
"""

import os
import sys
from typing import Any, Dict

from fastapi import FastAPI, Request

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
_MODAL_DIR = os.path.join(_REPO_ROOT, "modal")
if _MODAL_DIR not in sys.path:
    sys.path.insert(0, _MODAL_DIR)

import webexec_runtime

_store = webexec_runtime.NamespaceStore()

app = FastAPI()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/")
async def execute_block(raw_request: Request) -> Dict[str, Any]:
    body = await raw_request.body()
    return webexec_runtime.handle_execute_block_request(
        store=_store,
        body=body,
        content_encoding=raw_request.headers.get("content-encoding"),
    )


# The websocket route needs msgpack; the HTTP transport works without it.
try:
    import msgpack  # noqa: F401

    webexec_runtime.register_ws_route(
        app=app,
        store=_store,
        max_connection_seconds=int(
            os.getenv("WEBEXEC_WS_MAX_CONNECTION_SECONDS", "3600")
        ),
        idle_timeout_seconds=int(os.getenv("WEBEXEC_WS_IDLE_TIMEOUT_SECONDS", "10")),
    )
except ImportError:
    pass
