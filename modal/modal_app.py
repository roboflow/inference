"""
Modal app definition for Custom Python Blocks web endpoint.

This module contains the Modal-specific code for executing untrusted user code
in sandboxes. It's separated from the main executor to avoid requiring Modal
as a dependency for the main inference package.

The actual execution/serialization logic lives in ``webexec_runtime.py``,
which is shipped into the Modal image via ``add_local_python_source`` so the
deployed container and the local test stub
(``tests/workflows/integration_tests/execution/local_webexec_app.py``) run
exactly the same code.
"""

import os
from typing import Any, Dict

from starlette.requests import Request

import modal
import webexec_runtime

WEBEXEC_MODAL_CLOUD = os.environ.get("WEBEXEC_MODAL_CLOUD", "aws")
WEBEXEC_MODAL_REGION = os.environ.get("WEBEXEC_MODAL_REGION", "us-east-1")
WEBEXEC_MODAL_ROUTING_REGION = os.environ.get("WEBEXEC_MODAL_ROUTING_REGION")

WEBEXEC_WS_MAX_CONNECTION_SECONDS = int(
    os.getenv("WEBEXEC_WS_MAX_CONNECTION_SECONDS", "3600")
)
WEBEXEC_WS_IDLE_TIMEOUT_SECONDS = int(
    os.getenv("WEBEXEC_WS_IDLE_TIMEOUT_SECONDS", "10")
)

# Deploy-time configuration.
#
# The executor app name stays fixed at ``webexec``. Cloud / region env vars
# still control where that single executor is deployed.

app = modal.App("webexec")


INFERENCE_VERSION = os.getenv("INFERENCE_VERSION")
WEBEXEC_INFERENCE_DOCKER_IMAGE = os.getenv("WEBEXEC_INFERENCE_DOCKER_IMAGE", "roboflow/roboflow-inference-server-cpu")


def get_inference_image():
    """Get the Modal Image for inference."""

    # Use the pre-built shared image or create on-the-fly
    global INFERENCE_VERSION
    if not INFERENCE_VERSION:
        try:
            from inference.core.version import __version__

            INFERENCE_VERSION = __version__
        except ImportError:
            INFERENCE_VERSION = "latest"

    image = (
        modal.Image.from_registry(f"{WEBEXEC_INFERENCE_DOCKER_IMAGE}:{INFERENCE_VERSION}")
        .apt_install(
            "libgl1-mesa-glx",
            "libglib2.0-0",
            "libgomp1",
            "libsm6",
            "libxext6",
            "libxrender-dev",
            "ffmpeg",
            "wget",
        )
        .pip_install(inference_version)
        .pip_install("fastapi[standard]", "msgpack")
        # Ship the shared runtime from the repo checkout so the container runs
        # the same code as the local test stub, independent of the pinned
        # inference version installed above.
        .add_local_python_source("webexec_runtime")
    )
    return image


_executor_decorator_kwargs = {
    "image": get_inference_image(),
    "restrict_modal_access": True,  # Restrict Modal access for security
    "timeout": 700,
    "enable_memory_snapshot": True,  # Enable memory snapshotting for faster cold starts
    "scaledown_window": 60,
    "cloud": WEBEXEC_MODAL_CLOUD,
    "region": WEBEXEC_MODAL_REGION,
    "buffer_containers": 1,
    "env": {
        "WEBEXEC_WS_MAX_CONNECTION_SECONDS": str(WEBEXEC_WS_MAX_CONNECTION_SECONDS),
        "WEBEXEC_WS_IDLE_TIMEOUT_SECONDS": str(WEBEXEC_WS_IDLE_TIMEOUT_SECONDS),
    },
}
if WEBEXEC_MODAL_ROUTING_REGION:
    _executor_decorator_kwargs["routing_region"] = WEBEXEC_MODAL_ROUTING_REGION


@app.cls(**_executor_decorator_kwargs)
@modal.concurrent(max_inputs=10)
class Executor:
    """Parameterized Modal class for executing custom Python blocks via web endpoint."""

    # Parameterize by workspace_id
    workspace_id: str = modal.parameter()

    @modal.enter()
    def identify(self):
        print(f"Initializing sandbox for {self.workspace_id}")
        self._store = webexec_runtime.NamespaceStore()

    def _get_store(self) -> webexec_runtime.NamespaceStore:
        store = getattr(self, "_store", None)
        if store is None:
            store = webexec_runtime.NamespaceStore()
            self._store = store
        return store

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def execute_block(self, raw_request: Request) -> Dict[str, Any]:
        """Execute the custom block with the given inputs via web endpoint.

        Accepts plain JSON or gzip-compressed JSON (Content-Encoding: gzip).

        Returns:
            Dictionary with results or error information
        """
        body = await raw_request.body()
        return webexec_runtime.handle_execute_block_request(
            store=self._get_store(),
            body=body,
            content_encoding=raw_request.headers.get("content-encoding"),
        )

    @modal.asgi_app(requires_proxy_auth=True)
    def wsapp(self):
        """Expose a FastAPI sub-application with a WebSocket route.

        Each binary frame is a msgpack dict with the same fields as the HTTP
        request (``code_str``, ``imports``, ``run_function_name``, ``inputs``).

        Images arrive as raw JPEG ``bytes`` (no base64), keyed under
        ``_jpeg_bytes`` inside image dicts.  The response is also msgpack.
        """
        return webexec_runtime.build_wsapp(
            store=self._get_store(),
            max_connection_seconds=WEBEXEC_WS_MAX_CONNECTION_SECONDS,
            idle_timeout_seconds=WEBEXEC_WS_IDLE_TIMEOUT_SECONDS,
        )
