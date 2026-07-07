"""Local stand-in for the Modal ``webexec`` app used by integration tests.

Serves the exact same execution code as production: it imports
``webexec_runtime`` from the inference package (the module ``modal/modal_app.py``
runs in the Modal container) and only adds the transport glue Modal normally
provides — the HTTP POST route, the ``/ws`` websocket app, and a ``/health``
probe. Tests point ``MODAL_WEB_ENDPOINT_URL`` / ``MODAL_WS_ENDPOINT_URL`` at
this server (see the ``local_webexec_server`` fixture in ``conftest.py``).

Run with:
    uvicorn local_webexec_app:app --port <port>
"""

import os
from typing import Any, Dict

from fastapi import FastAPI, Request

from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
    webexec_runtime,
)

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
