"""
Modal executor for Custom Python Blocks in Workflows using Web Endpoints.

This module handles the execution of untrusted user code in Modal sandboxes
using web endpoints for better security and no size limitations.

Two transport modes are available, controlled by ``WEBEXEC_TRANSPORT``:

* **http** (default) — JSON POST with gzip compression and persistent
  ``requests.Session``.
* **websocket** — persistent WebSocket connection with msgpack binary frames.
  Eliminates per-request HTTP overhead and base64 image encoding.

Performance notes
-----------------
- A persistent ``requests.Session`` (http) or ``websocket`` (ws) connection
  is reused across frames.
- Images are re-encoded at ``WEBEXEC_JPEG_QUALITY`` (default 75).
- In websocket mode, images are sent as raw JPEG bytes inside msgpack
  (no base64), saving ~33 % payload size and CPU.
- User code is shipped only on the first frame for a given code hash. The
  server caches the compiled namespace keyed by hash, so subsequent frames
  send ``code_hash`` only and skip server-side ``compile()``/``exec()``.
  On cache miss (container restart, HTTP load-balanced to a new replica)
  the server returns ``UnknownCodeHash`` and the client retries once with
  the full code.
"""

import base64
import gzip
import hashlib
import json
import os
import threading
import time as _time
from typing import Any, Dict, Optional

import numpy as np
import requests

from inference.core.env import (
    MODAL_ANONYMOUS_WORKSPACE_NAME,
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
)
from inference.core.logger import logger
from inference.core.workflows.errors import DynamicBlockCodeError, DynamicBlockError
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils import (
    build_traceback_string,
    extract_code_snippet,
)
from inference.core.workflows.prototypes.block import BlockResult

# Check if Modal credentials are available
if MODAL_TOKEN_ID and MODAL_TOKEN_SECRET:
    MODAL_AVAILABLE = True
else:
    MODAL_AVAILABLE = False
    logger.info("Modal credentials not configured")

from datetime import datetime

from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_detections_kind,
    deserialize_image_kind,
    deserialize_video_metadata_kind,
)


def _compute_code_hash(code_str: str, imports: Optional[list]) -> str:
    """Stable hash for a python block's code + imports.

    Must match ``Executor._get_code_hash`` in ``modal/modal_app.py`` so the
    server can look up a previously-cached compiled namespace when the client
    sends only ``code_hash`` instead of the full ``code_str``.
    """
    content = (code_str or "") + "\n" + "\n".join(imports or [])
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def _serialise_image_for_webexec(image: Any) -> dict:
    """Encode an image at the webexec JPEG quality (default 75) instead of the
    WorkflowImageData default of 95.  For a 1080p frame this saves ~60-70 %
    of the base64 payload with negligible visual loss in a preview scenario."""
    from inference.core.env import WEBEXEC_JPEG_QUALITY
    from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
    from inference.core.workflows.execution_engine.entities.base import (
        ParentOrigin,
    )

    numpy_image = image.numpy_image
    b64 = base64.b64encode(
        encode_image_to_jpeg_bytes(numpy_image, jpeg_quality=WEBEXEC_JPEG_QUALITY)
    ).decode("ascii")

    result: Dict[str, Any] = {
        "type": "base64",
        "value": b64,
        "video_metadata": image.video_metadata.dict() if image.video_metadata else None,
    }

    parent_metadata = image.parent_metadata
    root_metadata = image.workflow_root_ancestor_metadata
    if parent_metadata.parent_id != root_metadata.parent_id:
        result["parent_id"] = parent_metadata.parent_id
        result["parent_origin"] = ParentOrigin.from_origin_coordinates_system(
            parent_metadata.origin_coordinates
        ).model_dump()
        result["root_parent_id"] = root_metadata.parent_id
        result["root_parent_origin"] = ParentOrigin.from_origin_coordinates_system(
            root_metadata.origin_coordinates
        ).model_dump()

    return result


def serialize_for_modal_remote_execution(inputs: Dict[str, Any]) -> str:
    from datetime import datetime

    import numpy as np

    class InputJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return {"_type": "datetime", "value": obj.isoformat()}
            elif isinstance(obj, bytes):
                return {
                    "_type": "bytes",
                    "value": base64.b64encode(obj).decode("utf-8"),
                }
            elif isinstance(obj, np.ndarray):
                return {
                    "_type": "ndarray",
                    "value": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
            elif hasattr(obj, "__dict__"):
                return {
                    "_type": "object",
                    "class": obj.__class__.__name__,
                    "value": str(obj),
                }
            return super().default(obj)

    def patch_for_modal_serialization(value):
        import supervision as sv

        from inference.core.workflows.core_steps.common.serializers import (
            serialise_sv_detections,
            serialize_video_metadata_kind,
        )
        from inference.core.workflows.execution_engine.entities.base import (
            VideoMetadata,
            WorkflowImageData,
        )

        if isinstance(value, sv.Detections):
            serialized = serialise_sv_detections(detections=value)
            serialized["_type"] = "sv_detections"
        elif isinstance(value, WorkflowImageData):
            serialized = _serialise_image_for_webexec(value)
            serialized["_type"] = "workflow_image"
        elif isinstance(value, VideoMetadata):
            serialized = serialize_video_metadata_kind(value)
            serialized["_type"] = "video_metadata"
        elif isinstance(value, dict):
            serialized = {
                k: patch_for_modal_serialization(v) if k != "_type" else v
                for k, v in value.items()
            }
        elif isinstance(value, list):
            serialized = [patch_for_modal_serialization(item) for item in value]
        else:
            serialized = value

        return serialized

    serialized_inputs = {}
    for key, value in inputs.items():
        serialized_inputs[key] = patch_for_modal_serialization(value)

    return json.dumps(serialized_inputs, cls=InputJSONEncoder)


def deserialize_for_modal_remote_execution(json_str: str) -> BlockResult:
    def decode_inputs(obj):
        """Decode from modal remote execution."""
        # datetime is already imported at the top level

        if isinstance(obj, dict):
            # Check for special type markers
            if "_type" in obj:
                if obj["_type"] == "datetime":
                    return datetime.fromisoformat(obj["value"])
                elif obj["_type"] == "bytes":
                    return base64.b64decode(obj["value"])
                elif obj["_type"] == "ndarray":
                    arr = np.array(obj["value"], dtype=obj["dtype"])
                    return arr.reshape(obj["shape"])
                elif obj["_type"] == "object":
                    return obj["value"]
                elif obj["_type"] == "sv_detections":
                    # First decode any nested special types in the dict
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_detections_kind("input", decoded_obj)
                elif obj["_type"] == "video_metadata":
                    # First decode any nested special types
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_video_metadata_kind("input", decoded_obj)
                elif obj["_type"] == "workflow_image":
                    # First decode any nested special types
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_image_kind("input", decoded_obj)

            # TODO: Not sure we actually need this anymore?
            # For backward compatibility, check if this is a WorkflowImageData without _type marker
            if obj.get("type") == "base64" and "value" in obj and "_type" not in obj:
                # Decode nested datetimes first
                if "video_metadata" in obj and obj["video_metadata"]:
                    obj["video_metadata"] = decode_inputs(obj["video_metadata"])
                return deserialize_image_kind("input", obj)

            # Recursively process dict values
            return {k: decode_inputs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decode_inputs(item) for item in obj]
        else:
            return obj

    # Parse and decode inputs
    parsed_inputs = json.loads(json_str)
    inputs = decode_inputs(parsed_inputs)
    return inputs


class ModalExecutor:
    """Manages execution of Custom Python Blocks in Modal sandboxes via web endpoints.

    Optimizations over a naive per-frame ``requests.post``:
    * **Persistent session** – reuses TCP + TLS connection across frames.
    * **Lower JPEG quality** – images are re-encoded at ``WEBEXEC_JPEG_QUALITY``
      (default 75 vs 95) to shrink payloads significantly.
    * **Executor caching** – a single instance is kept per workspace in
      ``block_scaffolding._MODAL_EXECUTOR_CACHE`` so the session survives
      across frames.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None
        # Hashes we believe the server has cached. HTTP requests can land on
        # any replica so a miss may still happen; we retry with full code.
        self._known_code_hashes: set = set()

    def _get_session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Modal-Key": MODAL_TOKEN_ID,
                    "Modal-Secret": MODAL_TOKEN_SECRET,
                }
            )
        return self._session

    def _get_endpoint_url(self, workspace_id: str) -> str:
        if self._base_url is None:
            env_url = os.environ.get("MODAL_WEB_ENDPOINT_URL")
            if env_url:
                self._base_url = env_url
            else:
                workspace = MODAL_WORKSPACE_NAME
                app_name = "webexec"
                class_name = "executor"
                method_name = "execute-block"

                label = f"{app_name}-{class_name}-{method_name}"
                if len(label) > 56:
                    hash_str = hashlib.sha256(label.encode()).hexdigest()[:6]
                    label = f"{label[:56]}-{hash_str}"

                self._base_url = f"https://{workspace}--{label}.modal.run"

        return f"{self._base_url}?workspace_id={workspace_id}"

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> BlockResult:
        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor | credentials_check",
            )

        workspace = workspace_id if workspace_id else self.workspace_id

        try:
            endpoint_url = self._get_endpoint_url(workspace)

            inputs_json = serialize_for_modal_remote_execution(inputs)

            code_hash = _compute_code_hash(
                python_code.run_function_code or "",
                python_code.imports,
            )

            if (
                not workspace
                or workspace == "anonymous"
                or workspace == "unauthorized"
                or workspace == MODAL_ANONYMOUS_WORKSPACE_NAME
            ):
                from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION

                if not MODAL_ALLOW_ANONYMOUS_EXECUTION:
                    raise DynamicBlockError(
                        public_message="Modal validation requires an API key when anonymous execution is disabled. "
                        "Please provide an API key or enable anonymous execution by setting "
                        "MODAL_ALLOW_ANONYMOUS_EXECUTION=True",
                        context="modal_executor | validation_authentication",
                    )

            # Hash-only path: skip shipping ``code_str`` and ``imports`` when
            # we believe the server already has this hash cached. On a miss
            # the server returns ``UnknownCodeHash`` and we resend full code.
            send_full_code = code_hash not in self._known_code_hashes
            result = self._post_execute(
                endpoint_url=endpoint_url,
                python_code=python_code,
                inputs_json=inputs_json,
                code_hash=code_hash,
                send_full_code=send_full_code,
            )

            if (
                not send_full_code
                and not result.get("success", False)
                and result.get("error_type") == "UnknownCodeHash"
            ):
                # Server replica doesn't have this hash cached; retry once.
                self._known_code_hashes.discard(code_hash)
                result = self._post_execute(
                    endpoint_url=endpoint_url,
                    python_code=python_code,
                    inputs_json=inputs_json,
                    code_hash=code_hash,
                    send_full_code=True,
                )

            if result.get("success", False):
                self._known_code_hashes.add(code_hash)

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                error_type = result.get("error_type", "RuntimeError")
                line_number = result.get("line_number")
                function_name = result.get("function_name") or "run"
                code = python_code.run_function_code

                message = (
                    f"Error in line {line_number}, in {function_name}: {error_type}: {error_msg}"
                    if line_number
                    else f"{error_type}: {error_msg}"
                )

                code_snippet = None
                traceback_str = None
                if line_number and code:
                    snippet = extract_code_snippet(code, line_number)
                    code_snippet = snippet.lstrip("\n") if snippet else None
                    traceback_str = build_traceback_string(
                        code, line_number, function_name, error_type, error_msg
                    )

                raise DynamicBlockCodeError(
                    public_message=message,
                    block_type_name=block_type_name,
                    error_line=line_number,
                    code_snippet=code_snippet,
                    traceback_str=traceback_str,
                    stdout=result.get("stdout"),
                    stderr=result.get("stderr"),
                )

            json_result = result.get("result", "{}")
            return deserialize_for_modal_remote_execution(json_result)

        except requests.exceptions.RequestException as e:
            raise DynamicBlockError(
                public_message=f"Failed to connect to Modal endpoint: {str(e)}",
                context="modal_executor | http_connection",
            )

    def _post_execute(
        self,
        endpoint_url: str,
        python_code: PythonCode,
        inputs_json: str,
        code_hash: str,
        send_full_code: bool,
    ) -> Dict[str, Any]:
        """Build the gzip-JSON request and POST it. Returns the parsed JSON.

        When ``send_full_code`` is False we omit ``code_str`` and ``imports``;
        the server uses ``code_hash`` to locate its cached compiled namespace.
        """
        request_payload: Dict[str, Any] = {
            "code_hash": code_hash,
            "run_function_name": python_code.run_function_name,
            "inputs_json": inputs_json,
        }
        if send_full_code:
            request_payload["code_str"] = python_code.run_function_code
            request_payload["imports"] = python_code.imports or []

        body_bytes = json.dumps(request_payload).encode("utf-8")
        compressed = gzip.compress(body_bytes, compresslevel=1)

        session = self._get_session()
        response = session.post(
            endpoint_url,
            data=compressed,
            timeout=30,
            headers={
                "Content-Type": "application/json",
                "Content-Encoding": "gzip",
            },
        )

        if response.status_code != 200:
            raise DynamicBlockError(
                public_message=f"Modal endpoint returned status {response.status_code}: {response.text}",
                context="modal_executor | http_request",
            )

        return response.json()


def validate_code_in_modal(
    python_code: PythonCode, workspace_id: Optional[str] = None
) -> bool:
    """Validate Python code syntax in a Modal sandbox via web endpoint.

    Args:
        python_code: The Python code to validate
        workspace_id: The workspace ID for Modal App

    Returns:
        True if code is valid, raises otherwise

    Raises:
        DynamicBlockError: If code validation fails
    """
    # Check if Modal is available
    if not MODAL_AVAILABLE:
        raise DynamicBlockError(
            public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
            context="modal_executor | credentials_check",
        )

    workspace = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME

    # Construct the full code to validate (same as in create_dynamic_module)
    full_code = python_code.run_function_code
    if python_code.init_function_code:
        full_code += "\n\n" + python_code.init_function_code

    # Escape the code for safe embedding in the validation function
    # Use repr() to properly escape quotes and special characters
    escaped_code = repr(full_code)

    # Simple validation code that checks syntax
    validation_code = PythonCode(
        type="PythonCode",
        imports=[],
        run_function_code=f"""
import ast

def validate_syntax():
    try:
        # Try to compile the user code
        code = {escaped_code}
        compile(code, "<string>", "exec")
        # Try to parse as AST to check structure
        ast.parse(code)
        return {{"valid": True}}
    except SyntaxError as e:
        return {{"valid": False, "error": str(e), "line": e.lineno}}
    except Exception as e:
        return {{"valid": False, "error": str(e)}}
""",
        run_function_name="validate_syntax",
        init_function_code=None,
        init_function_name="init",
    )

    executor = ModalExecutor(workspace_id=workspace)

    try:
        # For validation, we don't need complex inputs, just pass empty JSON
        result = executor.execute_remote(
            block_type_name="validation",
            python_code=validation_code,
            inputs={},
            workspace_id=workspace,
        )

        if result.get("valid") is False:
            error_msg = result.get("error", "Unknown syntax error")
            line_no = result.get("line", None)
            if line_no:
                error_msg = f"Line {line_no}: {error_msg}"
            raise DynamicBlockError(
                public_message=f"Code validation failed: {error_msg}",
                context="modal_executor | code_validation",
            )

        return True

    except Exception as e:
        if isinstance(e, DynamicBlockError):
            raise
        raise DynamicBlockError(
            public_message=f"Code validation failed: {str(e)}",
            context="modal_executor | code_validation",
        )


# ======================================================================
# WebSocket + msgpack transport (optional, enabled via WEBEXEC_TRANSPORT)
# ======================================================================


def _serialize_image_for_msgpack(image: Any) -> dict:
    """Encode a WorkflowImageData as a dict with raw JPEG bytes (no base64)."""
    from inference.core.env import WEBEXEC_JPEG_QUALITY
    from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
    from inference.core.workflows.execution_engine.entities.base import ParentOrigin

    jpeg_bytes: bytes = encode_image_to_jpeg_bytes(
        image.numpy_image,
        jpeg_quality=WEBEXEC_JPEG_QUALITY,
    )

    result: Dict[str, Any] = {
        "_type": "workflow_image",
        "_jpeg_bytes": jpeg_bytes,
        "parent_id": image.parent_metadata.parent_id,
    }
    if image.video_metadata:
        result["video_metadata"] = {
            "_type": "video_metadata",
            **image.video_metadata.dict(),
        }

    parent_metadata = image.parent_metadata
    root_metadata = image.workflow_root_ancestor_metadata
    if parent_metadata.parent_id != root_metadata.parent_id:
        result["parent_origin"] = ParentOrigin.from_origin_coordinates_system(
            parent_metadata.origin_coordinates
        ).model_dump()
        result["root_parent_id"] = root_metadata.parent_id
        result["root_parent_origin"] = ParentOrigin.from_origin_coordinates_system(
            root_metadata.origin_coordinates
        ).model_dump()

    return result


def serialize_inputs_for_msgpack(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Convert workflow inputs to a msgpack-friendly dict.

    Images become ``{"_type": "workflow_image", "_jpeg_bytes": <bytes>, ...}``.
    Detections and other tagged types keep their ``_type`` markers but remain
    plain dicts/lists so msgpack can handle them.
    """
    import supervision as sv

    from inference.core.workflows.core_steps.common.serializers import (
        serialise_sv_detections,
        serialize_video_metadata_kind,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

    def _pack(value: Any) -> Any:
        if isinstance(value, sv.Detections):
            d = serialise_sv_detections(detections=value)
            d["_type"] = "sv_detections"
            return d
        if isinstance(value, WorkflowImageData):
            return _serialize_image_for_msgpack(value)
        if isinstance(value, VideoMetadata):
            d = serialize_video_metadata_kind(value)
            d["_type"] = "video_metadata"
            return d
        if isinstance(value, datetime):
            return {"_type": "datetime", "value": value.isoformat()}
        if isinstance(value, np.ndarray):
            return {
                "_type": "ndarray",
                "value": value.tolist(),
                "dtype": str(value.dtype),
                "shape": list(value.shape),
            }
        if isinstance(value, bytes):
            return value
        if isinstance(value, dict):
            return {k: _pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_pack(v) for v in value]
        return value

    return {k: _pack(v) for k, v in inputs.items()}


def _deserialize_msgpack_result(result: Any) -> Any:
    """Inverse of ``_serialize_msgpack_result`` on the server side."""
    if isinstance(result, dict):
        _type = result.get("_type")
        if _type == "datetime":
            return datetime.fromisoformat(result["value"])
        if _type == "ndarray":
            return np.array(result["value"], dtype=result["dtype"]).reshape(
                result["shape"]
            )
        return {k: _deserialize_msgpack_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_deserialize_msgpack_result(v) for v in result]
    return result


class WebSocketModalExecutor:
    """Executes Custom Python Blocks via a persistent WebSocket + msgpack.

    Falls back to HTTP (``ModalExecutor``) if the WebSocket connection
    cannot be established.

    Keep-alive strategy
    -------------------
    We intentionally do **not** ping on every frame: ``websocket-client``
    ``ping()`` is a synchronous write that adds work to every RTT. Instead we
    trust ``self._ws`` on the hot path and only reconnect on an actual
    ``send`` / ``recv`` failure. A lightweight daemon thread pings the
    connection when it has been idle for ``_KEEPALIVE_IDLE_SECONDS`` so NATs
    and proxies don't silently close the channel during quiet periods.
    """

    _KEEPALIVE_IDLE_SECONDS = 25.0

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        self._ws: Any = None
        self._ws_url: Optional[str] = None
        # Hashes already sent over the current WS connection. A single WS
        # is pinned to one container, so anything in this set is guaranteed
        # cached server-side until the connection drops.
        self._hashes_sent_on_ws: set = set()
        # Serialize hot-path send/recv with the keepalive ping so they don't
        # step on each other on the same socket.
        self._io_lock = threading.Lock()
        self._last_activity: float = 0.0
        self._keepalive_stop: Optional[threading.Event] = None
        self._keepalive_thread: Optional[threading.Thread] = None

    def _get_ws_url(self, workspace_id: str) -> str:
        if self._ws_url is not None:
            return self._ws_url

        env_url = os.environ.get("MODAL_WEB_ENDPOINT_URL", "")
        if env_url:
            base = env_url.rstrip("/")
        else:
            workspace = MODAL_WORKSPACE_NAME
            label = "webexec-executor-wsapp"
            if len(label) > 56:
                h = hashlib.sha256(label.encode()).hexdigest()[:6]
                label = f"{label[:56]}-{h}"
            base = f"https://{workspace}--{label}.modal.run"

        ws_base = base.replace("https://", "wss://").replace("http://", "ws://")
        self._ws_url = f"{ws_base}/ws?workspace_id={workspace_id}"
        return self._ws_url

    def _connect(self, workspace_id: str) -> None:
        import websocket as ws_lib

        url = self._get_ws_url(workspace_id)
        headers = {
            "Modal-Key": MODAL_TOKEN_ID,
            "Modal-Secret": MODAL_TOKEN_SECRET,
        }
        logger.info("[webexec-ws] Connecting to %s", url)
        self._ws = ws_lib.create_connection(
            url,
            header=[f"{k}: {v}" for k, v in headers.items()],
            timeout=30,
        )
        # New container -> no compiled namespaces cached yet.
        self._hashes_sent_on_ws = set()
        self._last_activity = _time.monotonic()
        self._ensure_keepalive_thread()
        logger.info("[webexec-ws] Connected")

    def _ensure_connection(self, workspace_id: str) -> None:
        # Hot path: trust the cached socket. A dead connection will surface
        # as an exception on the very next ``send``/``recv`` and we drop+
        # reconnect in the caller's except block (see ``_execute_ws``).
        if self._ws is None:
            self._connect(workspace_id)

    def _ensure_keepalive_thread(self) -> None:
        if self._keepalive_thread is not None and self._keepalive_thread.is_alive():
            return
        self._keepalive_stop = threading.Event()
        self._keepalive_thread = threading.Thread(
            target=self._keepalive_loop,
            args=(self._keepalive_stop,),
            name=f"webexec-ws-keepalive-{self.workspace_id}",
            daemon=True,
        )
        self._keepalive_thread.start()

    def _keepalive_loop(self, stop_event: threading.Event) -> None:
        """Ping the WS when the connection has been idle long enough.

        Skipped entirely while frames are flowing (``_last_activity`` is
        updated on every successful RTT). Uses ``acquire(blocking=False)`` so
        the keepalive never delays a real frame already in flight.
        """
        interval = self._KEEPALIVE_IDLE_SECONDS
        while not stop_event.wait(interval):
            ws = self._ws
            if ws is None:
                return
            idle = _time.monotonic() - self._last_activity
            if idle < interval:
                continue
            if not self._io_lock.acquire(blocking=False):
                # Frame in flight -> that's keepalive enough.
                continue
            try:
                ws = self._ws
                if ws is None:
                    return
                try:
                    ws.ping()
                    self._last_activity = _time.monotonic()
                    logger.debug("[webexec-ws] keepalive ping ok")
                except Exception as e:
                    logger.debug(
                        "[webexec-ws] keepalive ping failed (%s); dropping conn",
                        e,
                    )
                    try:
                        ws.close()
                    except Exception:
                        pass
                    self._ws = None
                    self._hashes_sent_on_ws = set()
                    return
            finally:
                self._io_lock.release()

    def close(self) -> None:
        """Best-effort teardown, mainly for tests."""
        if self._keepalive_stop is not None:
            self._keepalive_stop.set()
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
    ) -> BlockResult:
        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor | credentials_check",
            )

        workspace = workspace_id or self.workspace_id
        if not workspace or workspace in (
            "anonymous",
            "unauthorized",
            MODAL_ANONYMOUS_WORKSPACE_NAME,
        ):
            from inference.core.env import MODAL_ALLOW_ANONYMOUS_EXECUTION

            if not MODAL_ALLOW_ANONYMOUS_EXECUTION:
                raise DynamicBlockError(
                    public_message="Modal validation requires an API key when anonymous execution is disabled.",
                    context="modal_executor | validation_authentication",
                )

        try:
            import msgpack
        except ImportError:
            raise DynamicBlockError(
                public_message="WEBEXEC_TRANSPORT is set to 'websocket' but msgpack is not installed. "
                "Install it with: pip install msgpack",
                context="modal_executor | missing_dependency",
            )

        try:
            import websocket as _ws_lib  # noqa: F401
        except ImportError:
            raise DynamicBlockError(
                public_message="WEBEXEC_TRANSPORT is set to 'websocket' but websocket-client is not installed. "
                "Install it with: pip install websocket-client",
                context="modal_executor | missing_dependency",
            )

        return self._execute_ws(
            block_type_name,
            python_code,
            inputs,
            workspace,
            msgpack,
        )

    def _execute_ws(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace: str,
        msgpack: Any,
    ) -> BlockResult:
        t0 = _time.monotonic()

        packed_inputs = serialize_inputs_for_msgpack(inputs)
        t_ser = _time.monotonic()

        code_hash = _compute_code_hash(
            python_code.run_function_code or "",
            python_code.imports,
        )

        # Hash-only path: if we've already sent this code over the current WS
        # connection (pinned to one container), drop ``code_str`` + ``imports``
        # from every subsequent frame. The server looks up the cached
        # compiled namespace by hash.
        send_full_code = code_hash not in self._hashes_sent_on_ws

        frame_bytes = self._build_ws_frame(
            python_code=python_code,
            packed_inputs=packed_inputs,
            code_hash=code_hash,
            send_full_code=send_full_code,
            msgpack=msgpack,
        )
        t_pack = _time.monotonic()

        try:
            self._ensure_connection(workspace)
            with self._io_lock:
                self._ws.send_binary(frame_bytes)
                resp_bytes = self._ws.recv()
            self._last_activity = _time.monotonic()
        except Exception:
            self._ws = None
            self._hashes_sent_on_ws = set()
            raise

        t_rtt = _time.monotonic()

        result = msgpack.unpackb(resp_bytes, raw=False)

        # Fresh replica doesn't have this hash cached (can happen after a
        # reconnect or container restart). Retry once with full code.
        if (
            not send_full_code
            and not result.get("success", False)
            and result.get("error_type") == "UnknownCodeHash"
        ):
            self._hashes_sent_on_ws.discard(code_hash)
            logger.info(
                "[webexec-ws] server missed cached hash %s, resending full code",
                code_hash,
            )
            retry_frame = self._build_ws_frame(
                python_code=python_code,
                packed_inputs=packed_inputs,
                code_hash=code_hash,
                send_full_code=True,
                msgpack=msgpack,
            )
            try:
                self._ensure_connection(workspace)
                with self._io_lock:
                    self._ws.send_binary(retry_frame)
                    resp_bytes = self._ws.recv()
                self._last_activity = _time.monotonic()
            except Exception:
                self._ws = None
                self._hashes_sent_on_ws = set()
                raise
            result = msgpack.unpackb(resp_bytes, raw=False)

        if result.get("success", False):
            self._hashes_sent_on_ws.add(code_hash)

        t_done = _time.monotonic()

        logger.debug(
            "[webexec-ws-timing] serialize=%.0fms pack=%.0fms rtt=%.0fms unpack=%.0fms total=%.0fms bytes=%d hash_only=%s",
            (t_ser - t0) * 1000,
            (t_pack - t_ser) * 1000,
            (t_rtt - t_pack) * 1000,
            (t_done - t_rtt) * 1000,
            (t_done - t0) * 1000,
            len(frame_bytes),
            not send_full_code,
        )

        if not result.get("success", False):
            self._raise_code_error(result, block_type_name, python_code)

        return _deserialize_msgpack_result(result.get("result", {}))

    @staticmethod
    def _build_ws_frame(
        python_code: PythonCode,
        packed_inputs: Dict[str, Any],
        code_hash: str,
        send_full_code: bool,
        msgpack: Any,
    ) -> bytes:
        """Pack a msgpack frame, optionally omitting ``code_str``/``imports``.

        When ``send_full_code`` is False the server resolves the compiled
        namespace through its per-container cache keyed by ``code_hash``.
        """
        payload: Dict[str, Any] = {
            "code_hash": code_hash,
            "run_function_name": python_code.run_function_name,
            "inputs": packed_inputs,
        }
        if send_full_code:
            payload["code_str"] = python_code.run_function_code
            payload["imports"] = python_code.imports or []
        return msgpack.packb(payload, use_bin_type=True)

    @staticmethod
    def _raise_code_error(
        result: dict,
        block_type_name: str,
        python_code: PythonCode,
    ) -> None:
        error_msg = result.get("error", "Unknown error")
        error_type = result.get("error_type", "RuntimeError")
        line_number = result.get("line_number")
        function_name = result.get("function_name") or "run"
        code = python_code.run_function_code

        message = (
            f"Error in line {line_number}, in {function_name}: {error_type}: {error_msg}"
            if line_number
            else f"{error_type}: {error_msg}"
        )

        code_snippet = None
        traceback_str = None
        if line_number and code:
            snippet = extract_code_snippet(code, line_number)
            code_snippet = snippet.lstrip("\n") if snippet else None
            traceback_str = build_traceback_string(
                code,
                line_number,
                function_name,
                error_type,
                error_msg,
            )

        raise DynamicBlockCodeError(
            public_message=message,
            block_type_name=block_type_name,
            error_line=line_number,
            code_snippet=code_snippet,
            traceback_str=traceback_str,
            stdout=result.get("stdout"),
            stderr=result.get("stderr"),
        )


def get_modal_executor(workspace_id: Optional[str] = None) -> "ModalExecutor":
    """Returns the right executor based on ``WEBEXEC_TRANSPORT``."""
    from inference.core.env import WEBEXEC_TRANSPORT

    if WEBEXEC_TRANSPORT == "websocket":
        return WebSocketModalExecutor(workspace_id)  # type: ignore[return-value]
    return ModalExecutor(workspace_id)
