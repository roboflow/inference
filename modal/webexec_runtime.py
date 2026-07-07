"""Transport-agnostic runtime for the webexec Custom Python Blocks executor.

Single source of truth for the code that compiles and runs user blocks and
(de)serializes requests/responses. It is used by:

* ``modal/modal_app.py`` — the production Modal app (this module is shipped
  into the Modal image via ``add_local_python_source`` so the deployed
  container runs exactly this code), and
* ``tests/workflows/integration_tests/execution/local_webexec_app.py`` — the
  local stub server the integration tests run against.

This module must not import ``modal`` and must keep heavy dependencies
(numpy, fastapi, msgpack, inference serializers) inside function bodies:
it is imported at deploy time in an environment that only has the modal
client SDK installed.
"""

import base64
import gzip
import hashlib
import inspect
import json
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils import (
    capture_output,
)


class NoopDebugTraces:
    """No-op stand-in for the workflow-scoped ``debug_traces`` proxy.

    Debug traces rely on a ContextVar that is only bound in the local process
    that drives the run; it is never propagated into the executor. Without
    this stand-in, user code calling ``debug_traces.append(...)`` would raise
    ``NameError`` here even though it works locally. Injecting a no-op keeps
    the namespace consistent with local execution while silently discarding
    traces.
    """

    def append(self, *args, **kwargs) -> None:
        return None


def get_code_hash(code_str: str, imports: Optional[list]) -> str:
    """Compute a stable hash for the code to identify unique blocks.

    Must match ``_compute_code_hash`` in the client
    (``inference/.../dynamic_blocks/modal_executor.py``) so the server can look
    up a previously-cached compiled namespace when the client sends only
    ``code_hash`` instead of the full ``code_str``.
    """
    content = (code_str or "") + "\n" + "\n".join(imports if imports else [])
    return hashlib.md5(content.encode()).hexdigest()


class NamespaceStore:
    """Per-container cache of compiled block namespaces, keyed by code hash."""

    def __init__(self):
        self._code_namespaces: Dict[str, dict] = {}
        self._shared_globals: Dict[str, Any] = {}
        self._namespace_lock = threading.RLock()

    def get_cached(self, code_hash: str) -> Optional[dict]:
        namespace = self._code_namespaces.get(code_hash)
        if namespace is not None:
            return namespace
        with self._namespace_lock:
            return self._code_namespaces.get(code_hash)

    def get_or_initialize(
        self, code_hash: str, code_str: str, imports: list
    ) -> Tuple[Optional[dict], Optional[Dict[str, Any]]]:
        namespace = self._code_namespaces.get(code_hash)
        if namespace is not None:
            return namespace, None

        with self._namespace_lock:
            namespace = self._code_namespaces.get(code_hash)
            if namespace is not None:
                return namespace, None

            namespace = {
                "__name__": "__main__",
                "globals": self._shared_globals,
                # Mirror local execution, where block_scaffolding injects
                # `debug_traces` via IMPORTS_LINES. Here it is a no-op because
                # the debug trace ContextVar is not propagated into the sandbox.
                "debug_traces": NoopDebugTraces(),
            }
            import_code = "\n".join(imports) if imports else ""
            full_imports = f"""
from typing import Any, List, Dict, Set, Optional
import supervision as sv
import numpy as np
import math
import time
import json
import os
import requests
import cv2
import shapely
from inference.core.workflows.execution_engine.entities.base import Batch, WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult

{import_code}

from datetime import datetime
"""
            try:
                exec(full_imports, namespace)
                exec(code_str, namespace)
            except Exception as e:
                return None, {
                    "success": False,
                    "error": f"Code initialization failed: {str(e)}",
                    "error_type": type(e).__name__,
                }

            self._code_namespaces[code_hash] = namespace
            return namespace, None


def run_user_code(
    store: NamespaceStore,
    code_str: str,
    imports: list,
    run_function_name: str,
    inputs: dict,
    client_code_hash: str = "",
    workflow_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve the block namespace and execute the user's function.

    Two request modes are supported:
      1. Full code: ``code_str`` is present -> compute hash, compile if new.
      2. Hash-only: ``code_str`` is empty but ``client_code_hash`` is provided
         -> look up previously cached namespace; on miss return
         ``UnknownCodeHash`` so the client retries with the full code.

    Returns the response dict with a raw (unserialized) ``result`` value;
    callers apply the transport-specific result serialization.
    """
    if code_str:
        code_hash = get_code_hash(code_str, imports)
        namespace, error_response = store.get_or_initialize(
            code_hash=code_hash,
            code_str=code_str,
            imports=imports,
        )
        if error_response is not None:
            return error_response
    elif client_code_hash:
        code_hash = client_code_hash
        namespace = store.get_cached(code_hash)
        if namespace is None:
            return {
                "success": False,
                "error": (
                    f"Code not cached on this container for hash "
                    f"{code_hash}; client must resend full code."
                ),
                "error_type": "UnknownCodeHash",
                "code_hash": code_hash,
            }
    else:
        return {
            "success": False,
            "error": "Request must include either 'code_str' or 'code_hash'.",
            "error_type": "InvalidRequest",
        }

    if run_function_name not in namespace:
        return {
            "success": False,
            "error": f"Function '{run_function_name}' not found in code",
            "error_type": "NameError",
        }

    user_function = namespace[run_function_name]
    sig = inspect.signature(user_function)
    params = list(sig.parameters.keys())

    _workflow_context = workflow_context or {}

    try:
        with capture_output() as (stdout_buf, stderr_buf):
            # If function expects 'self' as first param, create a simple
            # object to pass
            if params and params[0] == "self":

                class BlockSelf:
                    def get_workflow_context(self) -> Dict[str, Any]:
                        return dict(_workflow_context)

                block_self = BlockSelf()
                result = user_function(block_self, **inputs)
            else:
                result = user_function(**inputs)

        return {
            "success": True,
            "result": result,
            "stdout": stdout_buf.getvalue() or None,
            "stderr": stderr_buf.getvalue() or None,
        }
    except Exception as e:
        resp: Dict[str, Any] = {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "stdout": stdout_buf.getvalue() or None,
            "stderr": stderr_buf.getvalue() or None,
        }
        # Get the line number and function name from evaluated code
        tb = traceback.extract_tb(e.__traceback__)
        if tb:
            frame = tb[-1]
            resp["line_number"] = frame.lineno
            resp["function_name"] = frame.name
        return resp


# ---------------------------------------------------------------------------
# Transport 1: HTTP + gzip JSON
# ---------------------------------------------------------------------------


def serialize_result_json(result: Any) -> str:
    """Serialize a user-code return value for the HTTP JSON transport.

    Note: this intentionally differs from the client's input serialization
    (which JPEG-compresses images); results use the lossless
    ``serialise_image`` encoding.
    """
    from datetime import datetime

    import numpy as np
    import supervision as sv

    from inference.core.workflows.core_steps.common.serializers import (
        serialise_image,
        serialise_sv_detections,
        serialize_video_metadata_kind,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

    class OutputJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return {"_type": "datetime", "value": obj.isoformat()}
            if isinstance(obj, bytes):
                return {
                    "_type": "bytes",
                    "value": base64.b64encode(obj).decode("utf-8"),
                }
            if isinstance(obj, np.ndarray):
                return {
                    "_type": "ndarray",
                    "value": obj.tolist(),
                    "dtype": str(obj.dtype),
                    "shape": obj.shape,
                }
            if hasattr(obj, "__dict__"):
                return {
                    "_type": "object",
                    "class": obj.__class__.__name__,
                    "value": str(obj),
                }
            return super().default(obj)

    def patch_for_modal_serialization(value):
        """Serialize value and add _type markers for client deserialization."""
        if isinstance(value, sv.Detections):
            serialized = serialise_sv_detections(detections=value)
            serialized["_type"] = "sv_detections"
        elif isinstance(value, WorkflowImageData):
            serialized = serialise_image(image=value)
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

    serialized_result = {
        k: patch_for_modal_serialization(v) for k, v in (result or {}).items()
    }
    return json.dumps(serialized_result, cls=OutputJSONEncoder)


def deserialize_inputs_json(json_str: str) -> dict:
    """Decode the client's ``inputs_json`` payload into Python objects.

    Mirrors ``deserialize_for_modal_remote_execution`` on the client side
    (``inference/.../dynamic_blocks/modal_executor.py``).
    """
    from datetime import datetime

    import numpy as np

    from inference.core.workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
        deserialize_image_kind,
        deserialize_video_metadata_kind,
    )

    def decode_inputs(obj):
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
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_detections_kind("input", decoded_obj)
                elif obj["_type"] == "video_metadata":
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_video_metadata_kind("input", decoded_obj)
                elif obj["_type"] == "workflow_image":
                    decoded_obj = {
                        k: decode_inputs(v) for k, v in obj.items() if k != "_type"
                    }
                    return deserialize_image_kind("input", decoded_obj)

            # TODO: Not sure we actually need this anymore?
            # For backward compatibility, check if this is a WorkflowImageData
            # without _type marker
            if obj.get("type") == "base64" and "value" in obj and "_type" not in obj:
                # Decode nested datetimes first
                if "video_metadata" in obj and obj["video_metadata"]:
                    obj["video_metadata"] = decode_inputs(obj["video_metadata"])
                return deserialize_image_kind("input", obj)

            return {k: decode_inputs(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decode_inputs(item) for item in obj]
        else:
            return obj

    return decode_inputs(json.loads(json_str))


def handle_execute_block_request(
    store: NamespaceStore,
    body: bytes,
    content_encoding: Optional[str] = None,
) -> Dict[str, Any]:
    """Handle one HTTP execute-block request body; returns the response dict.

    Accepts plain JSON or gzip-compressed JSON (Content-Encoding: gzip).
    """
    if content_encoding == "gzip":
        body = gzip.decompress(body)
    request = json.loads(body)

    code_str = request.get("code_str", "")
    imports = request.get("imports", [])
    run_function_name = request.get("run_function_name", "")
    inputs_json = request.get("inputs_json", "{}")
    client_code_hash = request.get("code_hash", "")
    workflow_context = request.get("workflow_context") or {}

    try:
        inputs = deserialize_inputs_json(inputs_json)

        response = run_user_code(
            store=store,
            code_str=code_str,
            imports=imports,
            run_function_name=run_function_name,
            inputs=inputs,
            client_code_hash=client_code_hash,
            workflow_context=workflow_context,
        )
        if response.get("success"):
            response["result"] = serialize_result_json(response["result"])
        return response
    except Exception as e:
        # Outer exception handler for non-execution errors (deserialization,
        # result serialization, etc.)
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
        }


# ---------------------------------------------------------------------------
# Transport 2: WebSocket + msgpack binary frames
# ---------------------------------------------------------------------------


def deserialize_msgpack_inputs(inputs_raw: dict) -> dict:
    """Convert msgpack-decoded input dict into Python objects."""
    import cv2
    import numpy as np

    from inference.core.workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
        deserialize_image_kind,
        deserialize_video_metadata_kind,
    )

    def _decode(obj):
        if isinstance(obj, dict):
            _type = obj.get("_type")

            if _type == "workflow_image" and "_jpeg_bytes" in obj:
                jpeg = obj["_jpeg_bytes"]
                arr = np.frombuffer(jpeg, dtype=np.uint8)
                numpy_image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                from inference.core.workflows.execution_engine.entities.base import (
                    ImageParentMetadata,
                    ParentOrigin,
                    WorkflowImageData,
                )

                video_metadata = None
                if obj.get("video_metadata"):
                    video_metadata = _decode(obj["video_metadata"])

                parent_id = obj.get("parent_id", "webexec")
                parent_origin = obj.get("parent_origin")
                root_parent_id = obj.get("root_parent_id")
                root_parent_origin = obj.get("root_parent_origin")

                parent_origin_coords = None
                if parent_origin:
                    parsed_origin = ParentOrigin.model_validate(parent_origin)
                    parent_origin_coords = parsed_origin.to_origin_coordinates_system()

                parent_metadata = ImageParentMetadata(
                    parent_id=parent_id,
                    origin_coordinates=parent_origin_coords,
                )

                workflow_root_ancestor_metadata = None
                if root_parent_id:
                    root_origin_coords = None
                    if root_parent_origin:
                        parsed_root_origin = ParentOrigin.model_validate(
                            root_parent_origin
                        )
                        root_origin_coords = (
                            parsed_root_origin.to_origin_coordinates_system()
                        )
                    workflow_root_ancestor_metadata = ImageParentMetadata(
                        parent_id=root_parent_id,
                        origin_coordinates=root_origin_coords,
                    )

                return WorkflowImageData(
                    parent_metadata=parent_metadata,
                    workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
                    numpy_image=numpy_image,
                    video_metadata=video_metadata,
                )

            if _type == "sv_detections":
                decoded = {k: _decode(v) for k, v in obj.items() if k != "_type"}
                return deserialize_detections_kind("input", decoded)
            if _type == "video_metadata":
                decoded = {k: _decode(v) for k, v in obj.items() if k != "_type"}
                return deserialize_video_metadata_kind("input", decoded)
            if _type == "workflow_image":
                decoded = {k: _decode(v) for k, v in obj.items() if k != "_type"}
                return deserialize_image_kind("input", decoded)
            if _type == "datetime":
                from datetime import datetime

                return datetime.fromisoformat(obj["value"])
            if _type == "ndarray":
                return np.array(obj["value"], dtype=obj["dtype"]).reshape(obj["shape"])
            if _type == "bytes":
                return (
                    base64.b64decode(obj["value"])
                    if isinstance(obj["value"], str)
                    else obj["value"]
                )

            return {k: _decode(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_decode(v) for v in obj]
        return obj

    return {k: _decode(v) for k, v in inputs_raw.items()}


def serialize_msgpack_result(result: Any) -> Any:
    """Serialize a user-code return value for msgpack transport.

    Mirrors the HTTP path's serialize_result_json logic, adding _type markers
    for WorkflowImageData, sv.Detections, and VideoMetadata so the client can
    reconstruct them.
    """
    from datetime import datetime

    import numpy as np
    import supervision as sv

    from inference.core.workflows.core_steps.common.serializers import (
        serialise_image,
        serialise_sv_detections,
        serialize_video_metadata_kind,
    )
    from inference.core.workflows.execution_engine.entities.base import (
        VideoMetadata,
        WorkflowImageData,
    )

    def _encode(obj):
        if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
            return obj
        if isinstance(obj, datetime):
            return {"_type": "datetime", "value": obj.isoformat()}
        if isinstance(obj, sv.Detections):
            serialized = serialise_sv_detections(detections=obj)
            serialized["_type"] = "sv_detections"
            return _encode(serialized)
        if isinstance(obj, WorkflowImageData):
            serialized = serialise_image(image=obj)
            serialized["_type"] = "workflow_image"
            return _encode(serialized)
        if isinstance(obj, VideoMetadata):
            serialized = serialize_video_metadata_kind(obj)
            serialized["_type"] = "video_metadata"
            return _encode(serialized)
        if isinstance(obj, np.ndarray):
            return {
                "_type": "ndarray",
                "value": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _encode(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_encode(v) for v in obj]
        return str(obj)

    return _encode(result)


def register_ws_route(
    app: Any,
    store: NamespaceStore,
    max_connection_seconds: int,
    idle_timeout_seconds: int,
) -> None:
    """Register the ``/ws`` execute route on a FastAPI app.

    Each binary frame is a msgpack dict with the same fields as the HTTP
    request (``code_str``, ``imports``, ``run_function_name``, ``inputs``).

    Images arrive as raw JPEG ``bytes`` (no base64), keyed under
    ``_jpeg_bytes`` inside image dicts. The response is also msgpack.
    """
    import asyncio

    import msgpack
    from fastapi import WebSocket, WebSocketDisconnect

    @app.websocket("/ws")
    async def ws_execute(websocket: WebSocket):
        await websocket.accept()
        connected_at = time.monotonic()
        try:
            while True:
                remaining = max_connection_seconds - (time.monotonic() - connected_at)
                if remaining <= 0:
                    await websocket.close(code=1000)
                    return
                try:
                    raw = await asyncio.wait_for(
                        websocket.receive_bytes(),
                        timeout=min(remaining, idle_timeout_seconds),
                    )
                except asyncio.TimeoutError:
                    await websocket.close(code=1000)
                    return
                request = msgpack.unpackb(raw, raw=False)

                code_str = request.get("code_str", "")
                imports = request.get("imports", [])
                run_function_name = request.get("run_function_name", "")
                inputs_raw = request.get("inputs", {})
                client_code_hash = request.get("code_hash", "")
                workflow_context = request.get("workflow_context") or {}

                inputs = deserialize_msgpack_inputs(inputs_raw)
                resp = await asyncio.to_thread(
                    run_user_code,
                    store,
                    code_str,
                    imports,
                    run_function_name,
                    inputs,
                    client_code_hash,
                    workflow_context,
                )

                if resp.get("success"):
                    resp["result"] = serialize_msgpack_result(resp["result"])

                await websocket.send_bytes(msgpack.packb(resp, use_bin_type=True))
        except WebSocketDisconnect:
            pass


def build_wsapp(
    store: NamespaceStore,
    max_connection_seconds: int,
    idle_timeout_seconds: int,
) -> Any:
    """Build the FastAPI sub-application exposing the WebSocket route."""
    from fastapi import FastAPI

    ws_app = FastAPI()
    register_ws_route(
        app=ws_app,
        store=store,
        max_connection_seconds=max_connection_seconds,
        idle_timeout_seconds=idle_timeout_seconds,
    )
    return ws_app
