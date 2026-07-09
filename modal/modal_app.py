"""
Modal app definition for Custom Python Blocks web endpoint.

This module contains the Modal-specific code for executing untrusted user code
in sandboxes. It's separated from the main executor to avoid requiring Modal
as a dependency for the main inference package.
"""

import asyncio
import base64
import gzip
import hashlib
import inspect
import json
import os
import threading
import time
import traceback
from typing import Any, Dict, Optional, Tuple

from starlette.requests import Request

import modal

from inference.core.env import WEBEXEC_INFERENCE_VERSION, WEBEXEC_MODAL_APP_NAME
from inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils import (
    capture_output,
)

WEBEXEC_MODAL_CLOUD = os.environ.get("WEBEXEC_MODAL_CLOUD", "aws")
WEBEXEC_MODAL_REGION = os.environ.get("WEBEXEC_MODAL_REGION", "us-east-1")
WEBEXEC_MODAL_ROUTING_REGION = os.environ.get("WEBEXEC_MODAL_ROUTING_REGION")

WEBEXEC_WS_MAX_CONNECTION_SECONDS = int(
    os.getenv("WEBEXEC_WS_MAX_CONNECTION_SECONDS", "3600")
)
WEBEXEC_WS_IDLE_TIMEOUT_SECONDS = int(
    os.getenv("WEBEXEC_WS_IDLE_TIMEOUT_SECONDS", "10")
)
# Modal's ASGI data plane rejects websocket messages above ~2 MiB (it falls
# back to a blob upload that fails inside the container), so frames above this
# limit are split into a chunk-control frame plus raw chunks. Must match
# _WS_MAX_FRAME_BYTES in
# inference/core/workflows/execution_engine/v1/dynamic_blocks/modal_executor.py.
WEBEXEC_WS_MAX_FRAME_BYTES = 1024 * 1024


class _NoopDebugTraces:
    """No-op stand-in for the workflow-scoped ``debug_traces`` proxy.

    Debug traces rely on a ContextVar that is only bound in the local process
    that drives the run; it is never propagated into the Modal sandbox. Without
    this stand-in, user code calling ``debug_traces.append(...)`` would raise
    ``NameError`` here even though it works locally. Injecting a no-op keeps the
    namespace consistent with local execution while silently discarding traces.
    """

    def append(self, *args, **kwargs) -> None:
        return None


# Deploy-time configuration.
#
# The executor app name stays fixed at ``webexec``. Cloud / region env vars
# still control where that single executor is deployed.

# Create the Modal App
app = modal.App(WEBEXEC_MODAL_APP_NAME)


WEBEXEC_INFERENCE_DOCKER_IMAGE = os.getenv("WEBEXEC_INFERENCE_DOCKER_IMAGE", "roboflow/roboflow-inference-server-cpu")


def get_inference_image():
    """Get the Modal Image for inference."""

    # Use the pre-built shared image or create on-the-fly
    inference_version = WEBEXEC_INFERENCE_VERSION
    if not inference_version:
        try:
            from inference.core.version import __version__

            inference_version = __version__
        except ImportError:
            inference_version = "latest"

    image = (
        modal.Image.from_registry(f"{WEBEXEC_INFERENCE_DOCKER_IMAGE}:{inference_version}")
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
        .pip_install("fastapi[standard]", "msgpack")  # Add FastAPI for web endpoints
        .entrypoint([])
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

    # Store state for each unique code block within this container
    # Key is the hash of the code, value is the namespace dict for that code
    _code_namespaces: Dict[str, dict] = {}

    # Shared globals dict that all custom python blocks can access
    _shared_globals: Dict[str, Any] = {}

    @modal.enter()
    def identify(self):
        print(f"Initializing sandbox for {self.workspace_id}")
        # Initialize the namespaces dict and shared globals
        self._code_namespaces = {}
        self._shared_globals = {}
        self._namespace_lock = threading.RLock()

    def _get_code_hash(self, code_str: str, imports: list) -> str:
        """Compute a stable hash for the code to identify unique blocks."""
        # Combine code and imports to create a unique identifier
        content = code_str + "\n" + "\n".join(imports if imports else [])
        return hashlib.md5(content.encode()).hexdigest()

    def _get_namespace_lock(self) -> threading.RLock:
        namespace_lock = getattr(self, "_namespace_lock", None)
        if namespace_lock is None:
            namespace_lock = threading.RLock()
            self._namespace_lock = namespace_lock
        return namespace_lock

    def _get_cached_namespace(self, code_hash: str) -> Optional[dict]:
        namespace = self._code_namespaces.get(code_hash)
        if namespace is not None:
            return namespace
        with self._get_namespace_lock():
            return self._code_namespaces.get(code_hash)

    def _get_or_initialize_namespace(
        self, code_hash: str, code_str: str, imports: list
    ) -> Tuple[Optional[dict], Optional[Dict[str, Any]]]:
        namespace = self._code_namespaces.get(code_hash)
        if namespace is not None:
            return namespace, None

        with self._get_namespace_lock():
            namespace = self._code_namespaces.get(code_hash)
            if namespace is not None:
                return namespace, None

            namespace = {
                "__name__": "__main__",
                "globals": self._shared_globals,
                # Mirror local execution, where block_scaffolding injects
                # `debug_traces` via IMPORTS_LINES. Here it is a no-op because
                # the debug trace ContextVar is not propagated into the sandbox.
                "debug_traces": _NoopDebugTraces(),
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

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    async def execute_block(self, raw_request: Request) -> Dict[str, Any]:
        """Execute the custom block with the given inputs via web endpoint.

        Accepts plain JSON or gzip-compressed JSON (Content-Encoding: gzip).

        Returns:
            Dictionary with results or error information
        """
        from datetime import datetime

        import numpy as np

        from inference.core.workflows.core_steps.common.deserializers import (
            deserialize_detections_kind,
            deserialize_image_kind,
            deserialize_video_metadata_kind,
        )

        body = await raw_request.body()
        if raw_request.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        request = json.loads(body)

        code_str = request.get("code_str", "")
        imports = request.get("imports", [])
        run_function_name = request.get("run_function_name", "")
        inputs_json = request.get("inputs_json", "{}")
        client_code_hash = request.get("code_hash")
        workflow_context = request.get("workflow_context") or {}

        # Resolve the effective code hash. Two request modes are supported:
        #   1. Full code: ``code_str`` is present -> compute hash, compile if new.
        #   2. Hash-only: ``code_str`` is empty but ``code_hash`` is provided ->
        #      look up previously cached namespace; on miss return
        #      ``UnknownCodeHash`` so the client retries with the full code.
        if code_str:
            code_hash = self._get_code_hash(code_str, imports)
            namespace, error_response = self._get_or_initialize_namespace(
                code_hash=code_hash,
                code_str=code_str,
                imports=imports,
            )
            if error_response is not None:
                return error_response
        elif client_code_hash:
            code_hash = client_code_hash
            namespace = self._get_cached_namespace(code_hash)
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

        try:
            # we should import serialize_for_modal_remote_execution and deserialize_for_modal_remote_execution
            # from inference package, but need to have them included in the modal build for that
            # so just copy pasted for now
            from datetime import datetime

            from inference.core.workflows.core_steps.common.deserializers import (
                deserialize_detections_kind,
                deserialize_image_kind,
                deserialize_video_metadata_kind,
            )
            from inference.core.workflows.prototypes.block import BlockResult

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

                # Patch inputs with type markers for Modal serialization
                def patch_for_modal_serialization(value):
                    """Serialize value and add _type markers for Modal deserialization."""
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

                    # Apply standard serialization and add type markers based on original type
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
                        # Recursively process dict values
                        serialized = {
                            k: patch_for_modal_serialization(v) if k != "_type" else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, list):
                        # Recursively process list items
                        serialized = [
                            patch_for_modal_serialization(item) for item in value
                        ]
                    else:
                        serialized = value

                    return serialized

                serialized_inputs = {}
                for key, value in inputs.items():
                    serialized_inputs[key] = patch_for_modal_serialization(value)

                # Convert to JSON string
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
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_detections_kind("input", decoded_obj)
                            elif obj["_type"] == "video_metadata":
                                # First decode any nested special types
                                decoded_obj = {
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_video_metadata_kind(
                                    "input", decoded_obj
                                )
                            elif obj["_type"] == "workflow_image":
                                # First decode any nested special types
                                decoded_obj = {
                                    k: decode_inputs(v)
                                    for k, v in obj.items()
                                    if k != "_type"
                                }
                                return deserialize_image_kind("input", decoded_obj)

                        # TODO: Not sure we actually need this anymore?
                        # For backward compatibility, check if this is a WorkflowImageData without _type marker
                        if (
                            obj.get("type") == "base64"
                            and "value" in obj
                            and "_type" not in obj
                        ):
                            # Decode nested datetimes first
                            if "video_metadata" in obj and obj["video_metadata"]:
                                obj["video_metadata"] = decode_inputs(
                                    obj["video_metadata"]
                                )
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

            inputs = deserialize_for_modal_remote_execution(inputs_json)

            # Call the user function
            if run_function_name not in namespace:
                return {
                    "error": f"Function '{run_function_name}' not found in code",
                    "error_type": "NameError",
                }

            # Get the user's function
            user_function = namespace[run_function_name]

            # Check if function expects a 'self' parameter
            sig = inspect.signature(user_function)
            params = list(sig.parameters.keys())

            try:
                with capture_output() as (stdout_buf, stderr_buf):
                    # If function expects 'self' as first param, create a simple object to pass
                    if params and params[0] == "self":

                        class BlockSelf:
                            def get_workflow_context(self) -> Dict[str, Any]:
                                return dict(workflow_context)

                        block_self = BlockSelf()
                        result = user_function(block_self, **inputs)
                    else:
                        result = user_function(**inputs)

                json_result = serialize_for_modal_remote_execution(result)

                return {
                    "success": True,
                    "result": json_result,
                    "stdout": stdout_buf.getvalue() or None,
                    "stderr": stderr_buf.getvalue() or None,
                }
            except Exception as e:
                # On error, capture stdout/stderr and return error details
                result = {
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
                    result["line_number"] = frame.lineno
                    result["function_name"] = frame.name

                return result

        except Exception as e:
            # Outer exception handler for non-execution errors (deserialization, etc.)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # ------------------------------------------------------------------
    # Transport 2: WebSocket + msgpack binary frames (opt-in)
    # ------------------------------------------------------------------

    @staticmethod
    def _run_user_code_ws(
        executor: Any,
        code_str: str,
        imports: list,
        run_function_name: str,
        inputs: dict,
        client_code_hash: str = "",
        workflow_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute user code for the WebSocket transport.

        When ``code_str`` is empty but ``client_code_hash`` is provided, look up
        a previously cached namespace on this container. On cache miss we return
        ``UnknownCodeHash`` so the client resends the full code once.
        """
        if code_str:
            code_hash = executor._get_code_hash(code_str, imports)
            namespace, error_response = executor._get_or_initialize_namespace(
                code_hash=code_hash,
                code_str=code_str,
                imports=imports,
            )
            if error_response is not None:
                return error_response
        elif client_code_hash:
            code_hash = client_code_hash
            namespace = executor._get_cached_namespace(code_hash)
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
            tb = traceback.extract_tb(e.__traceback__)
            if tb:
                frame = tb[-1]
                resp["line_number"] = frame.lineno
                resp["function_name"] = frame.name
            return resp

    @staticmethod
    def _deserialize_msgpack_inputs(inputs_raw: dict) -> dict:
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
                        parent_origin_coords = (
                            parsed_origin.to_origin_coordinates_system()
                        )

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
                    return np.array(obj["value"], dtype=obj["dtype"]).reshape(
                        obj["shape"]
                    )
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

    @staticmethod
    def _serialize_msgpack_result(result: Any) -> Any:
        """Serialize a user-code return value for msgpack transport.

        Mirrors the HTTP path's serialize_for_modal_remote_execution logic,
        adding _type markers for WorkflowImageData, sv.Detections, and
        VideoMetadata so the client can reconstruct them.
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

    @modal.asgi_app(requires_proxy_auth=True)
    def wsapp(self):
        """Expose a FastAPI sub-application with a WebSocket route.

        Each binary frame is a msgpack dict with the same fields as the HTTP
        request (``code_str``, ``imports``, ``run_function_name``, ``inputs``).

        Images arrive as raw JPEG ``bytes`` (no base64), keyed under
        ``_jpeg_bytes`` inside image dicts.  The response is also msgpack.
        """
        import msgpack
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect

        ws_app = FastAPI()

        executor_self = self

        @ws_app.websocket("/ws")
        async def ws_execute(websocket: WebSocket):
            await websocket.accept()
            connected_at = time.monotonic()
            try:
                while True:
                    remaining = WEBEXEC_WS_MAX_CONNECTION_SECONDS - (
                        time.monotonic() - connected_at
                    )
                    if remaining <= 0:
                        await websocket.close(code=1000)
                        return
                    try:
                        raw = await asyncio.wait_for(
                            websocket.receive_bytes(),
                            timeout=min(remaining, WEBEXEC_WS_IDLE_TIMEOUT_SECONDS),
                        )
                    except asyncio.TimeoutError:
                        await websocket.close(code=1000)
                        return
                    request = msgpack.unpackb(raw, raw=False)
                    if isinstance(request, dict) and "_chunked" in request:
                        parts = []
                        for _ in range(request["_chunked"]):
                            parts.append(
                                await asyncio.wait_for(
                                    websocket.receive_bytes(),
                                    timeout=WEBEXEC_WS_IDLE_TIMEOUT_SECONDS,
                                )
                            )
                        request = msgpack.unpackb(b"".join(parts), raw=False)

                    code_str = request.get("code_str", "")
                    imports = request.get("imports", [])
                    run_function_name = request.get("run_function_name", "")
                    inputs_raw = request.get("inputs", {})
                    client_code_hash = request.get("code_hash", "")
                    workflow_context = request.get("workflow_context") or {}

                    inputs = Executor._deserialize_msgpack_inputs(inputs_raw)
                    resp = await asyncio.to_thread(
                        Executor._run_user_code_ws,
                        executor_self,
                        code_str,
                        imports,
                        run_function_name,
                        inputs,
                        client_code_hash,
                        workflow_context,
                    )

                    if resp.get("success"):
                        resp["result"] = Executor._serialize_msgpack_result(
                            resp["result"]
                        )

                    payload = msgpack.packb(resp, use_bin_type=True)
                    if len(payload) > WEBEXEC_WS_MAX_FRAME_BYTES:
                        chunks = [
                            payload[i : i + WEBEXEC_WS_MAX_FRAME_BYTES]
                            for i in range(
                                0, len(payload), WEBEXEC_WS_MAX_FRAME_BYTES
                            )
                        ]
                        await websocket.send_bytes(
                            msgpack.packb(
                                {"_chunked": len(chunks)}, use_bin_type=True
                            )
                        )
                        for chunk in chunks:
                            await websocket.send_bytes(chunk)
                    else:
                        await websocket.send_bytes(payload)
            except WebSocketDisconnect:
                pass

        return ws_app
