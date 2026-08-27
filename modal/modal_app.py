"""
Modal app definition for Custom Python Blocks web endpoint.

This module contains the Modal-specific code for executing untrusted user code
in sandboxes. It's separated from the main executor to avoid requiring Modal
as a dependency for the main inference package.
"""

import asyncio
import base64
from contextlib import contextmanager
import gzip
import hashlib
import inspect
from io import StringIO
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, Generator, Optional, Tuple

from starlette.requests import Request

import modal

_thread_local = threading.local()
_install_lock = threading.Lock()


WEBEXEC_MODAL_APP_NAME = os.environ.get("WEBEXEC_MODAL_APP_NAME", "webexec")
WEBEXEC_MODAL_CLOUD = os.environ.get("WEBEXEC_MODAL_CLOUD", "aws")
WEBEXEC_MODAL_REGION = os.environ.get("WEBEXEC_MODAL_REGION", "us-east-1")
WEBEXEC_MODAL_ROUTING_REGION = os.environ.get("WEBEXEC_MODAL_ROUTING_REGION")

# NOTE: this cap and the protocol v2 session guarantee interact. Sessions are
# container-local and reconnects are not routed with any affinity, so every
# capped close of a STATEFUL run is likely to land on another container and
# surface as a (correct, but scheduled) session-lost failure roughly hourly.
# Making long stateful runs survive needs session-affine reconnect routing,
# externalized session state, or a drain/handoff before the cap — none of
# which exist yet. Until then, keep stateful custom Python runs shorter than
# this cap, or expect a checkpoint replay per hour.
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


class _TtlKeySet:
    """TTL'd, size-capped set of ids with refreshable timestamps.

    Backs the two container-local registries protocol v2 needs — which
    sessions this container holds runtime state for, and which request ids
    it has already started executing. Entries are kept in age order so
    pruning is an early-exit scan from the front.

    All access happens on the event loop thread, so no locking is needed.
    """

    def __init__(self, ttl_seconds: float, max_entries: int):
        from collections import OrderedDict

        self._ttl_seconds = ttl_seconds
        self._max_entries = max_entries
        self._seen: "OrderedDict[str, float]" = OrderedDict()

    def _prune(self) -> None:
        while len(self._seen) > self._max_entries:
            self._seen.popitem(last=False)
        deadline = time.monotonic() - self._ttl_seconds
        while self._seen:
            key, last_seen = next(iter(self._seen.items()))
            if last_seen >= deadline:
                break
            del self._seen[key]

    def add(self, key: str) -> None:
        if not key:
            return
        self._seen.pop(key, None)
        self._seen[key] = time.monotonic()
        self._prune()

    def refresh(self, key: str) -> bool:
        """Re-stamp an existing entry. Never adds. Returns whether present."""
        if not key or key not in self._seen:
            return False
        self.add(key)
        return True

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str) or not key:
            return False
        self._prune()
        return key in self._seen


class _WsResponseCache:
    """Bounded cache of request_id -> packed response (protocol v2 dedup).

    Lets a client safely resend a request whose response was lost to a
    dropped connection: an already-executed request returns its cached
    response instead of running the user code a second time.

    The retry window is seconds, so entries carry a short TTL; responses can
    embed serialized images, so eviction is byte-capped as well as
    entry-capped (LRU order). All access happens on the event loop thread,
    so no locking is needed.

    Retention is best effort — a single oversized response or a burst of
    concurrent large ones can evict an entry the client still wants. The
    at-most-once guarantee therefore does NOT rest on this cache: it rests
    on the executed-request registry (``_ws_executed``), which a cache miss
    falls back to so the resend gets a loud error instead of a second
    execution.
    """

    def __init__(
        self,
        max_entries: int = 128,
        max_bytes: int = 64 * 1024 * 1024,
        ttl_seconds: float = 120.0,
    ):
        from collections import OrderedDict

        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._ttl_seconds = ttl_seconds
        self._entries: "OrderedDict[str, Tuple[float, bytes]]" = OrderedDict()
        self._total_bytes = 0

    def _prune_expired(self) -> None:
        deadline = time.monotonic() - self._ttl_seconds
        while self._entries:
            key, (stored_at, payload) = next(iter(self._entries.items()))
            if stored_at >= deadline:
                break
            del self._entries[key]
            self._total_bytes -= len(payload)

    def get(self, key: Optional[str]) -> Optional[bytes]:
        if not key:
            return None
        self._prune_expired()
        entry = self._entries.get(key)
        if entry is None:
            return None
        # Re-stamp on hit so LRU order stays age-sorted for the
        # early-exit expiry scan above.
        self._entries[key] = (time.monotonic(), entry[1])
        self._entries.move_to_end(key)
        return entry[1]

    def put(self, key: str, payload: bytes) -> None:
        old = self._entries.pop(key, None)
        if old is not None:
            self._total_bytes -= len(old[1])
        self._entries[key] = (time.monotonic(), payload)
        self._total_bytes += len(payload)
        self._prune_expired()
        while self._entries and (
            len(self._entries) > self._max_entries
            or self._total_bytes > self._max_bytes
        ):
            _, (_, evicted) = self._entries.popitem(last=False)
            self._total_bytes -= len(evicted)


# mirrors inference/core/workflows/execution_engine/v1/dynamic_blocks (avoiding `from inference import ...`)
class _ThreadDispatchStream:
    """Stream wrapper that tees writes into a per-thread StringIO buffer
    (when one is active) while still forwarding them to the original stream.

    Threads that are not capturing see normal stdout/stderr behaviour; threads
    that are capturing get both: the buffer keeps an in-memory copy for error
    payloads, and the original stream still receives the bytes so ``print()``
    output continues to reach Docker / the process stdout.
    """

    def __init__(self, original, attr_name: str):
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_attr_name", attr_name)

    def _get_buffer(self):
        return getattr(_thread_local, self._attr_name, None)

    def write(self, data):
        buf = self._get_buffer()
        if buf is not None:
            try:
                buf.write(data)
            except Exception:
                pass
        return self._original.write(data)

    def flush(self):
        buf = self._get_buffer()
        if buf is not None:
            try:
                buf.flush()
            except Exception:
                pass
        return self._original.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return self._original.isatty()

    def __getattr__(self, name):
        return getattr(self._original, name)


def _install_dispatchers() -> None:
    if isinstance(sys.stdout, _ThreadDispatchStream):
        return
    with _install_lock:
        if isinstance(sys.stdout, _ThreadDispatchStream):
            return
        sys.stdout = _ThreadDispatchStream(sys.stdout, "_capture_stdout")
        sys.stderr = _ThreadDispatchStream(sys.stderr, "_capture_stderr")


@contextmanager
def capture_output() -> Generator[Tuple[StringIO, StringIO], None, None]:
    """Context manager to capture stdout and stderr for the current thread.

    Uses per-thread buffers via ``threading.local`` so concurrent calls in
    different threads capture independently without any global lock.
    """
    _install_dispatchers()
    stdout_buf, stderr_buf = StringIO(), StringIO()
    _thread_local._capture_stdout = stdout_buf
    _thread_local._capture_stderr = stderr_buf
    try:
        yield stdout_buf, stderr_buf
    finally:
        _thread_local._capture_stdout = None
        _thread_local._capture_stderr = None


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


INFERENCE_VERSION = os.getenv("INFERENCE_VERSION")
WEBEXEC_INFERENCE_DOCKER_IMAGE = os.getenv(
    "WEBEXEC_INFERENCE_DOCKER_IMAGE", "roboflow/roboflow-inference-server-cpu"
)


def get_inference_image():
    """Get the Modal Image for inference."""

    # Use the pre-built shared image or create on-the-fly
    inference_version = INFERENCE_VERSION
    if not inference_version:
        try:
            from inference.core.version import __version__

            inference_version = __version__
        except ImportError:
            inference_version = "latest"

    image = (
        modal.Image.from_registry(
            f"{WEBEXEC_INFERENCE_DOCKER_IMAGE}:{inference_version}"
        )
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
        # FastAPI serves the web endpoints; msgpack is a hard requirement of
        # the websocket transport (wsapp crash-loops on boot without it on
        # base images that predate msgpack in inference's requirements).
        .pip_install("fastapi[standard]", "msgpack")
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
        import uuid

        print(f"Initializing sandbox for {self.workspace_id}")
        # Initialize the namespaces dict and shared globals
        self._code_namespaces = {}
        self._shared_globals = {}
        self._namespace_lock = threading.RLock()
        # Protocol v2 websocket state (see wsapp): all touched only on the
        # event loop thread.
        self._container_id = uuid.uuid4().hex
        self._ws_sessions = _TtlKeySet(
            ttl_seconds=self._WS_SESSION_TTL_SECONDS,
            max_entries=self._WS_SESSION_MAX_ENTRIES,
        )
        # Request ids whose user code this container has STARTED running.
        # Entries are tiny, so this outlives the response cache by a wide
        # margin: it is what makes execution at-most-once even when the
        # response payload is gone.
        self._ws_executed = _TtlKeySet(
            ttl_seconds=self._WS_EXECUTED_TTL_SECONDS,
            max_entries=self._WS_EXECUTED_MAX_ENTRIES,
        )
        self._ws_response_cache = _WsResponseCache()
        self._ws_inflight: Dict[str, "asyncio.Task"] = {}

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

    # A session id is registered only once user code has SUCCESSFULLY
    # executed for it on this container — mirroring the client's
    # ``_had_success`` latch. Registering earlier (e.g. at hello time)
    # would make a failed "session lost" handshake mark the session as
    # known, so the very next reconnect would silently pass the check
    # this mechanism exists to fail. An ALREADY known session is refreshed
    # on every hello, so a long-lived but currently failing stream cannot
    # age out of the registry while its namespaces are still here.
    _WS_SESSION_TTL_SECONDS = 7200.0
    _WS_SESSION_MAX_ENTRIES = 4096
    # Request ids live only as long as a client could still resend one:
    # 3 attempts with reconnects, bounded by the client's read timeout.
    _WS_EXECUTED_TTL_SECONDS = 3600.0
    _WS_EXECUTED_MAX_ENTRIES = 32768

    def _ws_session_seen(self, session_id: str) -> bool:
        """Whether user code for this session already ran on this container.

        Answers the client's hello: ``session_known=False`` on a reconnect
        tells the client its Python runtime state (mutated globals in the
        cached namespaces) lives in some other, gone container — the client
        fails loudly instead of silently continuing with reset state.

        A hit also re-stamps the entry: the namespaces this answer is about
        are never evicted, so the registry must not expire under a session
        that is still actively connecting.
        """
        return self._ws_sessions.refresh(session_id)

    def _ws_register_session(self, session_id: str) -> None:
        self._ws_sessions.add(session_id)

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
            # Mirror of block_scaffolding's tensor-native IMPORTS_LINES extension.
            # Guarded import: this runs inside the sandbox on the PINNED inference
            # image — releases predating the tensor pivot lack the constant (and
            # releases that carry it keep the extension a no-op unless the sandbox
            # env enables the flag, which today it never does; tensor_native+modal
            # is additionally blocked at compile time). Importing the constant
            # instead of copy-pasting the lines keeps the two lists drift-free.
            try:
                from inference.core.env import ENABLE_TENSOR_DATA_REPRESENTATION
                from inference.core.workflows.execution_engine.v1.dynamic_blocks.block_scaffolding import (
                    TENSOR_NATIVE_IMPORTS_LINES,
                )

                tensor_native_imports = (
                    "\n".join(TENSOR_NATIVE_IMPORTS_LINES)
                    if ENABLE_TENSOR_DATA_REPRESENTATION
                    else ""
                )
            except ImportError:
                tensor_native_imports = ""
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
{tensor_native_imports}

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

        async def send_payload(websocket: WebSocket, payload: bytes) -> None:
            """Send one logical response, chunking oversized payloads."""
            if len(payload) > WEBEXEC_WS_MAX_FRAME_BYTES:
                chunks = [
                    payload[i : i + WEBEXEC_WS_MAX_FRAME_BYTES]
                    for i in range(0, len(payload), WEBEXEC_WS_MAX_FRAME_BYTES)
                ]
                await websocket.send_bytes(
                    msgpack.packb({"_chunked": len(chunks)}, use_bin_type=True)
                )
                for chunk in chunks:
                    await websocket.send_bytes(chunk)
            else:
                await websocket.send_bytes(payload)

        def _pack_ws_error(
            msgpack_module: Any,
            error_type: str,
            error: str,
            request_id: Optional[str],
        ) -> bytes:
            """Pack a minimal error response that can never fail to encode."""
            resp: Dict[str, Any] = {
                "success": False,
                "error": str(error),
                "error_type": str(error_type),
            }
            if request_id:
                resp["request_id"] = request_id
            return msgpack_module.packb(resp, use_bin_type=True)

        async def execute_request(
            request: dict, request_id: Optional[str], session_id: str
        ) -> bytes:
            """Run one execution request and return the packed response.

            Registered in ``_ws_inflight`` while running so a resend of the
            same ``request_id`` (from this or another connection) awaits the
            original execution instead of running the user code a second
            time; the completed payload then moves to the response cache for
            resends that arrive after completion.

            Never raises: user code is at-most-once, so once execution has
            started every outcome — including a result this server cannot
            serialize — has to come back as a packed response that is also
            cached. An escaping exception would kill the connection without
            a cache entry and invite the client to resend a request whose
            side effects already happened.
            """
            payload: Optional[bytes] = None
            try:
                code_str = request.get("code_str", "")
                imports = request.get("imports", [])
                run_function_name = request.get("run_function_name", "")
                inputs_raw = request.get("inputs", {})
                client_code_hash = request.get("code_hash", "")
                workflow_context = request.get("workflow_context") or {}

                inputs = Executor._deserialize_msgpack_inputs(inputs_raw)
                # Mark BEFORE running: from here on a resend must never
                # re-execute, however this call ends.
                if request_id:
                    executor_self._ws_executed.add(request_id)
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
                    resp["result"] = Executor._serialize_msgpack_result(resp["result"])
                    # Only now has runtime state actually been built here;
                    # see the note on _ws_register_session.
                    executor_self._ws_register_session(session_id)

                if request_id:
                    resp["request_id"] = request_id

                payload = msgpack.packb(resp, use_bin_type=True)
            except Exception as error:
                traceback.print_exc()
                payload = _pack_ws_error(
                    msgpack,
                    error_type=type(error).__name__,
                    error=(
                        "The custom Python block ran, but the server could not "
                        f"return its response: {error}"
                    ),
                    request_id=request_id,
                )
            finally:
                # Cache before clearing in-flight so a resend arriving in
                # between finds one or the other, never a gap. On
                # cancellation payload stays None: nothing to cache, but the
                # executed marker already bars a re-run.
                if request_id:
                    if payload is not None:
                        executor_self._ws_response_cache.put(request_id, payload)
                    executor_self._ws_inflight.pop(request_id, None)
            return payload

        @ws_app.websocket("/ws")
        async def ws_execute(websocket: WebSocket):
            await websocket.accept()
            connected_at = time.monotonic()
            conn_session_id = ""
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

                    # ---- protocol v2 control frames ----
                    kind = request.get("_kind") if isinstance(request, dict) else None
                    if kind == "hello":
                        conn_session_id = request.get("session_id") or ""
                        await websocket.send_bytes(
                            msgpack.packb(
                                {
                                    "_kind": "hello",
                                    "proto": 2,
                                    "idle_timeout_s": WEBEXEC_WS_IDLE_TIMEOUT_SECONDS,
                                    "session_known": executor_self._ws_session_seen(
                                        conn_session_id
                                    ),
                                    # Lets the client tell whether a
                                    # reconnect landed on the same container
                                    # (whose dedup cache it may rely on) or
                                    # a different one.
                                    "container_id": executor_self._container_id,
                                },
                                use_bin_type=True,
                            )
                        )
                        continue
                    if kind == "heartbeat":
                        # An application-level frame is the only thing that
                        # resets this loop's receive_bytes idle timer;
                        # protocol ws pings are consumed by the ASGI layer
                        # and never reach here.
                        await websocket.send_bytes(
                            msgpack.packb({"_kind": "heartbeat_ack"}, use_bin_type=True)
                        )
                        continue

                    request_id = (
                        request.get("request_id") if isinstance(request, dict) else None
                    )
                    if request_id:
                        cached_payload = executor_self._ws_response_cache.get(
                            request_id
                        )
                        if cached_payload is not None:
                            # Resend of a request already executed (the
                            # client lost the response): answer from cache,
                            # never run user code twice.
                            await send_payload(websocket, cached_payload)
                            continue
                        # No await between the in-flight lookup and the
                        # insert below, so two connections can't both start
                        # the same request.
                        task = executor_self._ws_inflight.get(request_id)
                        if task is None:
                            if request_id in executor_self._ws_executed:
                                # Already ran here, but its response is no
                                # longer available (evicted, or the call was
                                # cancelled). Re-running would duplicate the
                                # block's side effects, so fail loudly
                                # instead — execution stays at-most-once.
                                await send_payload(
                                    websocket,
                                    _pack_ws_error(
                                        msgpack,
                                        error_type="ResponseNoLongerAvailable",
                                        error=(
                                            "This request already executed on "
                                            "this container and its response is "
                                            "no longer available. It was not "
                                            "run again, to avoid duplicating "
                                            "the block's side effects."
                                        ),
                                        request_id=request_id,
                                    ),
                                )
                                continue
                            task = asyncio.create_task(
                                execute_request(request, request_id, conn_session_id)
                            )
                            executor_self._ws_inflight[request_id] = task
                        # shield: this connection dying must not cancel an
                        # execution another connection may be waiting on
                        # (or will resend for).
                        payload = await asyncio.shield(task)
                    else:
                        payload = await execute_request(request, None, conn_session_id)
                    await send_payload(websocket, payload)
            except WebSocketDisconnect:
                pass

        return ws_app
