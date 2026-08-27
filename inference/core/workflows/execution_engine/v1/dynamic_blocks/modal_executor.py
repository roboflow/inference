"""
Modal executor for Custom Python Blocks in Workflows using Web Endpoints.

This module handles the execution of untrusted user code in Modal sandboxes
using web endpoints for better security and no size limitations.

Two transport modes are available, controlled by ``WEBEXEC_TRANSPORT``:

* **http** — JSON POST with gzip compression and persistent ``requests.Session``.
* **websocket** (default) — persistent WebSocket connections with msgpack binary
  frames. Eliminates per-request HTTP overhead and base64 image encoding.

"""

import base64
import gzip
import hashlib
import json
import os
import sys
import threading
import time as _time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import requests

from inference.core.env import (
    MODAL_ANONYMOUS_WORKSPACE_NAME,
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
    WEBEXEC_JPEG_QUALITY,
    WEBEXEC_MODAL_APP_NAME,
    WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS,
    WEBEXEC_WS_CONNECTION_POOL_SIZE,
    WEBEXEC_WS_IDLE_RELEASE_SECONDS,
    WEBEXEC_WS_READ_TIMEOUT_SECONDS,
)
from inference.core.logger import logger
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.core_steps.common.serializers import (
    serialize_video_metadata_kind,
)
from inference.core.workflows.errors import DynamicBlockCodeError, DynamicBlockError
from inference.core.workflows.execution_engine.entities.base import ParentOrigin
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

_WEBEXEC_EXECUTOR_CLASS_LABEL = "executor"
_WEBEXEC_HTTP_METHOD_LABEL = "execute-block"
_WEBEXEC_WS_METHOD_LABEL = "wsapp"

# Modal's ASGI data plane rejects websocket messages above ~2 MiB (it falls
# back to a blob upload that fails inside the container), so frames above this
# limit are split into a chunk-control frame plus raw chunks. Must match
# WEBEXEC_WS_MAX_FRAME_BYTES in modal/modal_app.py.
_WS_MAX_FRAME_BYTES = 1024 * 1024


def _split_ws_frames(frame_bytes: bytes, msgpack: Any) -> list:
    """Split an oversized frame into a chunk-control frame plus raw chunks."""
    if len(frame_bytes) <= _WS_MAX_FRAME_BYTES:
        return [frame_bytes]
    chunks = [
        frame_bytes[i : i + _WS_MAX_FRAME_BYTES]
        for i in range(0, len(frame_bytes), _WS_MAX_FRAME_BYTES)
    ]
    return [msgpack.packb({"_chunked": len(chunks)}, use_bin_type=True), *chunks]


def _build_webexec_endpoint_base(method_label: str) -> str:
    workspace = MODAL_WORKSPACE_NAME
    app_name = WEBEXEC_MODAL_APP_NAME
    label = f"{app_name}-{_WEBEXEC_EXECUTOR_CLASS_LABEL}-{method_label}"
    if len(label) > 56:
        hash_str = hashlib.sha256(label.encode()).hexdigest()[:6]
        label = f"{label[:56]}-{hash_str}"
    return f"https://{workspace}--{label}.modal.run"


def _coerce_http_endpoint_to_ws_endpoint(endpoint_url: str) -> str:
    """Map a legacy execute_block endpoint URL to the wsapp endpoint URL.

    Operates on the subdomain label only, so it survives regional hosts
    (e.g. ``...-execute-block.eu-west.modal.run``).
    """
    parts = urlsplit(endpoint_url.rstrip("/"))
    subdomain, dot, rest = parts.netloc.partition(".")
    if subdomain.endswith(f"-{_WEBEXEC_HTTP_METHOD_LABEL}"):
        subdomain = (
            subdomain[: -len(_WEBEXEC_HTTP_METHOD_LABEL)] + _WEBEXEC_WS_METHOD_LABEL
        )
    netloc = subdomain + dot + rest
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))


def _as_ws_endpoint_url(endpoint_url: str, workspace_id: str) -> str:
    parts = urlsplit(endpoint_url.rstrip("/"))
    scheme = parts.scheme.replace("https", "wss", 1).replace("http", "ws", 1)
    path = parts.path.rstrip("/")
    if not path.endswith("/ws"):
        path = f"{path}/ws"
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["workspace_id"] = workspace_id
    return urlunsplit(
        (
            scheme,
            parts.netloc,
            path,
            urlencode(query),
            parts.fragment,
        )
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
    """Encode an image at the configured WEBEXEC_JPEG_QUALITY (default 95)."""
    numpy_image = image.numpy_image
    b64 = base64.b64encode(
        encode_image_to_jpeg_bytes(numpy_image, jpeg_quality=WEBEXEC_JPEG_QUALITY)
    ).decode("ascii")

    video_metadata = None
    if image.video_metadata:
        video_metadata = serialize_video_metadata_kind(image.video_metadata)

    result: Dict[str, Any] = {
        "type": "base64",
        "value": b64,
        "video_metadata": video_metadata,
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


def _raise_on_unconverted_tensor_native_value(obj: Any) -> None:
    """Defense-in-depth for the Modal wire (tensor pivot, D2-REVISED / Step 7).

    Under ``ENABLE_TENSOR_DATA_REPRESENTATION`` the dynamic-block representation
    boundary converts ``legacy_compatibility`` inputs BEFORE remote execution and
    ``tensor_native`` is compile-blocked on modal, so no native
    ``inference_models`` object (nor a bare ``torch.Tensor``) should ever reach
    this serializer. If one does, the generic ``__dict__`` fallback would
    silently stringify it on the wire — raise loudly instead. Provably inert
    when the flag is off (guard first; natives cannot exist flag-off anyway),
    keeping flag-off behavior byte-identical. Lazy imports mirror this file's
    style; the modules are already in ``sys.modules`` on any code path that
    reaches Modal serialization (the block scaffolding imports the boundary).
    """
    # The boundary module's import-time constant is the established patch point
    # for tests; reading it through the module keeps the two arms in lockstep.
    from inference.core.workflows.execution_engine.v1.dynamic_blocks import (
        representation_boundary,
    )

    if not representation_boundary._TENSOR_REPRESENTATION_ACTIVE:
        return
    import torch

    from inference_models.models.base.classification import (
        ClassificationPrediction,
        MultiLabelClassificationPrediction,
    )
    from inference_models.models.base.instance_segmentation import InstanceDetections
    from inference_models.models.base.keypoints_detection import KeyPoints
    from inference_models.models.base.object_detection import Detections

    if not isinstance(
        obj,
        (
            Detections,
            InstanceDetections,
            KeyPoints,
            ClassificationPrediction,
            MultiLabelClassificationPrediction,
            torch.Tensor,
        ),
    ):
        return
    offending_type = type(obj)
    raise representation_boundary.RepresentationBoundaryError(
        public_message=(
            f"A native tensor object of type "
            f"`{offending_type.__module__}.{offending_type.__qualname__}` reached the "
            f"Modal wire serializer unconverted. The representation boundary converts "
            f"`legacy_compatibility` inputs before remote execution (and "
            f"`tensor_native` is not executable over modal), so this indicates the "
            f"value bypassed the boundary. Declare the input's kind on the dynamic "
            f"block so the boundary can convert it — refusing to fall back to generic "
            f"object serialization, which would silently corrupt the value on the wire."
        ),
        context="workflow_execution | modal_executor | input_serialization",
        offending_type=offending_type,
    )


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
                # Native tensor objects also carry __dict__ — refuse to
                # stringify them (see the helper's docstring); arbitrary
                # non-native objects keep the pre-existing generic contract.
                _raise_on_unconverted_tensor_native_value(obj)
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
    """Manages execution of Custom Python Blocks in Modal sandboxes via web endpoints."""

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        self._base_url: Optional[str] = None
        self._session: Optional[requests.Session] = None
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
                self._base_url = _build_webexec_endpoint_base(
                    method_label=_WEBEXEC_HTTP_METHOD_LABEL
                )

                # If we couldn't get it dynamically, construct it based on expected pattern
                if not self._base_url:
                    # URL pattern: https://{workspace}--{app}-{class}-{method_truncated}.modal.run
                    # Note: Modal truncates long labels to 63 chars with a hash suffix
                    workspace = MODAL_WORKSPACE_NAME
                    app_name = WEBEXEC_MODAL_APP_NAME
                    class_name = "executor"
                    method_name = "execute-block"

                    # The label would be: inference-custom-blocks-web-customblockexecutor-execute-block
                    # This is 62 chars, which might get truncated
                    label = f"{app_name}-{class_name}-{method_name}"
                    if (
                        len(label) > 56
                    ):  # Modal truncates at 56 chars and adds 7-char hash
                        import hashlib

                        hash_str = hashlib.sha256(label.encode()).hexdigest()[:6]
                        label = f"{label[:56]}-{hash_str}"

                    self._base_url = f"https://{workspace}--{label}.modal.run"

        # Add workspace_id as query parameter
        return f"{self._base_url}?workspace_id={workspace_id}"

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
        workflow_context: Optional[Dict[str, Any]] = None,
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
                workflow_context=workflow_context or {},
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
                    workflow_context=workflow_context or {},
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

            stdout = result.get("stdout")
            stderr = result.get("stderr")
            if stdout:
                sys.stdout.write(stdout)
                sys.stdout.flush()
            if stderr:
                sys.stderr.write(stderr)
                sys.stderr.flush()

            # Get the result and deserialize from JSON
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
        workflow_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build the gzip-JSON request and POST it. Returns the parsed JSON.

        When ``send_full_code`` is False we omit ``code_str`` and ``imports``;
        the server uses ``code_hash`` to locate its cached compiled namespace.
        """
        request_payload: Dict[str, Any] = {
            "code_hash": code_hash,
            "run_function_name": python_code.run_function_name,
            "inputs_json": inputs_json,
            "workflow_context": workflow_context,
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

    Validation intentionally uses the HTTP ``execute-block`` endpoint even when
    ``WEBEXEC_TRANSPORT=websocket`` for execution. Deployments that use
    websocket execution must keep both Modal methods deployed: ``execute-block``
    for validation and ``wsapp`` for execution.

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

    # Keep validation on HTTP. It is a control-plane check, while websocket is
    # only the execution fast path.
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


def _serialize_image_for_msgpack(image: Any) -> dict:
    """Encode a WorkflowImageData as a dict with raw JPEG bytes (no base64)."""
    from inference.core.env import WEBEXEC_JPEG_QUALITY
    from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
    from inference.core.workflows.core_steps.common.serializers import (
        serialize_video_metadata_kind,
    )
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
            **serialize_video_metadata_kind(image.video_metadata),
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
            return {k: _pack(v) for k, v in d.items()}
        if isinstance(value, WorkflowImageData):
            d = _serialize_image_for_msgpack(value)
            return {k: _pack(v) for k, v in d.items()}
        if isinstance(value, VideoMetadata):
            d = serialize_video_metadata_kind(value)
            d["_type"] = "video_metadata"
            return {k: _pack(v) for k, v in d.items()}
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
    """Inverse of ``_serialize_msgpack_result`` on the server side.

    Handles _type markers for datetime, ndarray, sv_detections,
    workflow_image, and video_metadata to mirror the HTTP path.
    """
    from inference.core.workflows.core_steps.common.deserializers import (
        deserialize_detections_kind,
        deserialize_image_kind,
        deserialize_video_metadata_kind,
    )

    if isinstance(result, dict):
        _type = result.get("_type")
        if _type == "datetime":
            return datetime.fromisoformat(result["value"])
        if _type == "ndarray":
            return np.array(result["value"], dtype=result["dtype"]).reshape(
                result["shape"]
            )
        if _type == "sv_detections":
            decoded = {
                k: _deserialize_msgpack_result(v)
                for k, v in result.items()
                if k != "_type"
            }
            return deserialize_detections_kind("result", decoded)
        if _type == "workflow_image":
            decoded = {
                k: _deserialize_msgpack_result(v)
                for k, v in result.items()
                if k != "_type"
            }
            return deserialize_image_kind("result", decoded)
        if _type == "video_metadata":
            decoded = {
                k: _deserialize_msgpack_result(v)
                for k, v in result.items()
                if k != "_type"
            }
            return deserialize_video_metadata_kind("result", decoded)
        return {k: _deserialize_msgpack_result(v) for k, v in result.items()}
    if isinstance(result, list):
        return [_deserialize_msgpack_result(v) for v in result]
    return result


class WebexecSessionLostError(Exception):
    """The server no longer holds this session's Python runtime state.

    Raised instead of silently reconnecting: a reconnect that lands on a
    fresh container would reset any state the block's code mutated across
    frames, producing wrong results with no error. Callers must fail the
    job (and replay from a checkpoint) rather than continue.
    """


class WebSocketModalExecutor:
    """Executes Custom Python Blocks via a persistent WebSocket + msgpack.

    Protocol v2 (negotiated in ``_connect``): every server message is a
    msgpack dict; ``_kind`` marks control frames (``hello``,
    ``heartbeat_ack``); execution requests carry a ``request_id`` the
    server echoes and dedups, making resend-after-failure safe. Against a
    v1 server the executor falls back to the legacy behavior.
    """

    # v1 fallback only; with a v2 server the interval derives from the
    # idle timeout the server advertises in the hello reply.
    _KEEPALIVE_IDLE_SECONDS = 25.0

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        self._ws: Any = None
        self._ws_url: Optional[str] = None
        self._hashes_sent_on_ws: set = set()
        self._io_lock = threading.Lock()
        self._last_activity: float = 0.0
        self._keepalive_stop: Optional[threading.Event] = None
        self._keepalive_thread: Optional[threading.Thread] = None
        self._session_id: str = uuid.uuid4().hex
        self._server_proto: int = 1
        self._server_idle_timeout: Optional[float] = None
        self._server_container_id: Optional[str] = None
        self._had_success: bool = False

    def _rotate_session(self, reason: str) -> None:
        """Start a fresh session after the current one's state is gone.

        Called when session loss is detected (so the loud failure happens
        exactly once, after which execution honestly restarts under a new
        session id) and when an idle connection is deliberately released.
        Without the rotation, the very next reconnect would find the old
        session id registered on some container and silently continue with
        reset runtime state — the exact bug the session check exists to
        prevent.
        """
        logger.info(
            "[webexec-ws] rotating session %s -> new session (%s)",
            self._session_id,
            reason,
        )
        self._session_id = uuid.uuid4().hex
        self._had_success = False
        self._hashes_sent_on_ws = set()

    def _get_ws_url(self, workspace_id: str) -> str:
        if self._ws_url is not None:
            return self._ws_url

        explicit_ws_url = os.environ.get("MODAL_WS_ENDPOINT_URL", "")
        if explicit_ws_url:
            base = explicit_ws_url.rstrip("/")
        else:
            legacy_http_url = os.environ.get("MODAL_WEB_ENDPOINT_URL", "")
            if legacy_http_url:
                base = _coerce_http_endpoint_to_ws_endpoint(legacy_http_url)
            else:
                base = _build_webexec_endpoint_base(
                    method_label=_WEBEXEC_WS_METHOD_LABEL
                )

        self._ws_url = _as_ws_endpoint_url(base, workspace_id)
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
            timeout=WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS,
        )
        self._ws.settimeout(WEBEXEC_WS_READ_TIMEOUT_SECONDS)
        # New container -> no compiled namespaces cached yet.
        self._hashes_sent_on_ws = set()
        try:
            self._handshake()
        except Exception:
            # Never leave a half-negotiated socket cached.
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            raise
        self._last_activity = _time.monotonic()
        self._ensure_keepalive_thread()
        logger.info(
            "[webexec-ws] Connected (proto=%d idle_timeout=%s)",
            self._server_proto,
            self._server_idle_timeout,
        )

    def _handshake(self) -> None:
        """Negotiate protocol v2; fall back to v1 on a legacy server.

        Sends a hello frame carrying this executor's session id. A v2
        server replies with its idle timeout (so the heartbeat interval is
        derived, not guessed) and whether this container still knows the
        session. A v1 server treats the hello as an execution request and
        replies without ``_kind``; that reply is discarded and the
        connection behaves as legacy.
        """
        import msgpack

        # Negotiate into locals and commit only at the end: the surviving
        # keepalive thread reads _server_proto/_server_idle_timeout
        # unlocked, and must not observe reset values mid-handshake (a v1
        # fallback interval of 25s outlasts a v2 server's 10s idle close).
        hello = msgpack.packb(
            {"_kind": "hello", "proto": 2, "session_id": self._session_id},
            use_bin_type=True,
        )
        self._ws.send_binary(hello)
        reply_raw = self._recv_bytes_frame()
        try:
            reply = msgpack.unpackb(reply_raw, raw=False)
        except Exception as error:
            raise ConnectionError(
                f"undecodable websocket handshake reply: {error}"
            ) from error
        if not (isinstance(reply, dict) and reply.get("_kind") == "hello"):
            # Legacy server: it executed the hello as an (empty) request and
            # returned its error response. Discard it and stay on v1.
            logger.info("[webexec-ws] server speaks protocol v1")
            self._server_proto = 1
            self._server_idle_timeout = None
            self._server_container_id = None
            return
        if self._had_success and not reply.get("session_known", False):
            # Rotate before raising so the failure is loud exactly once:
            # the next request starts an honest fresh session instead of
            # silently passing the check with reset runtime state.
            self._rotate_session("session lost on reconnect")
            raise WebexecSessionLostError(
                "The Modal container holding this custom Python session is "
                "gone; runtime state mutated by previous frames cannot be "
                "restored. Failing instead of silently continuing."
            )
        self._server_proto = 2
        self._server_idle_timeout = float(reply.get("idle_timeout_s") or 10.0)
        self._server_container_id = reply.get("container_id") or None

    def _ensure_connection(self, workspace_id: str) -> None:
        # Hot path: trust the cached socket. A dead connection will surface
        # as an exception on the very next ``send``/``recv`` and we drop+
        # reconnect in the caller's except block (see ``_execute_ws``).
        if self._ws is None:
            with self._io_lock:
                # Double-check inside the lock to prevent race where two
                # threads both see _ws as None and both call _connect(),
                # leaking a socket and keepalive thread.
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

        A connection idle past ``WEBEXEC_WS_IDLE_RELEASE_SECONDS`` is
        deliberately released instead of heartbeated forever: heartbeats
        keep the Modal container warm (and billed), so an abandoned
        connection would otherwise pin it for the full connection cap.
        Releasing rotates the session — runtime state is knowingly given
        up, so the next request starts fresh instead of failing loudly on
        a lost session.
        """
        while not stop_event.wait(self._heartbeat_interval()):
            ws = self._ws
            if ws is None:
                return
            # _last_activity tracks real frames only — a heartbeat must not
            # count as activity, or the idle-release below could never fire.
            idle = _time.monotonic() - self._last_activity
            if idle < self._heartbeat_interval():
                continue
            if not self._io_lock.acquire(blocking=False):
                # Frame in flight -> that's keepalive enough.
                continue
            try:
                ws = self._ws
                if ws is None:
                    return
                if idle >= WEBEXEC_WS_IDLE_RELEASE_SECONDS:
                    logger.info(
                        "[webexec-ws] connection idle for %.0fs; releasing it "
                        "(and its custom Python session) so the container "
                        "can scale down",
                        idle,
                    )
                    self._drop_ws_connection()
                    self._rotate_session("idle release")
                    return
                try:
                    self._send_heartbeat(ws)
                    logger.debug("[webexec-ws] keepalive ok")
                except Exception as e:
                    logger.debug(
                        "[webexec-ws] keepalive failed (%s); dropping conn",
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

    def _heartbeat_interval(self) -> float:
        """Heartbeat period, derived from the server's advertised idle timeout.

        Must be an application-level frame interval well under the server's
        ``receive_bytes`` timeout: protocol-level ws pings are answered by
        the ASGI layer and never reset that timer.
        """
        if self._server_proto == 2 and self._server_idle_timeout:
            return max(1.0, self._server_idle_timeout / 3.0)
        return self._KEEPALIVE_IDLE_SECONDS

    # A heartbeat ack is tiny and immediate; waiting the full read timeout
    # (sized for user-code execution) would let a half-open connection pin
    # _io_lock — and stall every real request — for minutes.
    _HEARTBEAT_ACK_TIMEOUT_SECONDS = 10.0

    def _send_heartbeat(self, ws: Any) -> None:
        """One heartbeat round-trip. Caller must hold ``_io_lock``."""
        import msgpack

        if self._server_proto != 2:
            # Legacy server: protocol ping is all v1 offers. It cannot reset
            # the server's app-level idle timer, but it does detect a dead
            # socket so the next request reconnects instead of crashing.
            ws.ping()
            return
        ws.send_binary(msgpack.packb({"_kind": "heartbeat"}, use_bin_type=True))
        ws.settimeout(self._HEARTBEAT_ACK_TIMEOUT_SECONDS)
        try:
            reply_raw = self._recv_bytes_frame(ws)
            try:
                reply = msgpack.unpackb(reply_raw, raw=False)
            except Exception as error:
                raise ConnectionError(
                    f"undecodable heartbeat reply: {error}"
                ) from error
        finally:
            ws.settimeout(WEBEXEC_WS_READ_TIMEOUT_SECONDS)
        if not (isinstance(reply, dict) and reply.get("_kind") == "heartbeat_ack"):
            raise ConnectionError(f"unexpected heartbeat reply: {reply!r}")

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

    def _drop_ws_connection(self) -> None:
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass
        self._ws = None
        self._hashes_sent_on_ws = set()

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
        workflow_context: Optional[Dict[str, Any]] = None,
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

        try:
            return self._execute_ws(
                block_type_name,
                python_code,
                inputs,
                workspace,
                msgpack,
                workflow_context or {},
            )
        except WebexecSessionLostError as error:
            raise DynamicBlockError(
                public_message=(
                    f"Custom Python session lost for block "
                    f"`{block_type_name}`: {error} The job must be retried "
                    "from its last checkpoint (e.g. re-run the chunk) so the "
                    "block's runtime state is rebuilt from the start."
                ),
                context="modal_executor | websocket_session_lost",
            ) from error

    def _send_recv_with_retry(
        self,
        frame_bytes: bytes,
        workspace: str,
        request_id: Optional[str] = None,
    ) -> bytes:
        """Send frame and receive response, reconnecting on failure.

        Against a v2 server every execution frame carries a ``request_id``
        the server dedups, so resending after a ``recv`` failure is safe —
        but only while the reconnect lands on the SAME container: the dedup
        cache is per-container, so a resend that reaches a different
        container would run the user code a second time. When the reconnect
        lands elsewhere the outcome is ambiguous and the request fails
        loudly instead.

        Against a v1 server (no dedup) the legacy rule holds: once
        ``send_binary`` succeeds the outcome is ambiguous and the frame is
        not resent.

        ``WebexecSessionLostError`` from the reconnect handshake propagates:
        state continuity is gone and the job must fail, not continue.
        """
        import msgpack

        frames = _split_ws_frames(frame_bytes, msgpack)
        last_exc: Optional[Exception] = None
        # Container the frame was accepted by on a previous attempt; set only
        # when a resend must be answered from that container's dedup cache.
        sent_to_container: Optional[str] = None
        attempts = 3
        for attempt in range(attempts):
            sent_ok = False
            try:
                self._ensure_connection(workspace)
                if (
                    sent_to_container is not None
                    and self._server_container_id != sent_to_container
                ):
                    raise DynamicBlockError(
                        public_message=(
                            "WebSocket connection to Modal endpoint lost after "
                            "the request was sent, and the reconnect reached a "
                            "different container. The custom Python block may "
                            "have already executed, so the frame was not "
                            "retried."
                        ),
                        context="modal_executor | websocket_response",
                    )
                # Hold the lock across send+recv so concurrent callers sharing
                # this executor's socket can't interleave a request/response.
                with self._io_lock:
                    container_at_send = self._server_container_id
                    for frame in frames:
                        self._ws.send_binary(frame)
                    sent_ok = True
                    resp_bytes = self._recv_reassembled(msgpack)
                self._last_activity = _time.monotonic()
                return resp_bytes
            except (WebexecSessionLostError, DynamicBlockError):
                raise
            except Exception as e:
                self._drop_ws_connection()
                resend_is_safe = self._server_proto == 2 and request_id
                if sent_ok and not resend_is_safe:
                    # v1: recv failed after the frame was sent; the remote may
                    # have already executed user code, so we don't resend and
                    # risk duplicate side effects.
                    logger.warning(
                        "[webexec-ws] response receive failed after frame was "
                        "sent; not retrying to avoid duplicate execution: %s",
                        e,
                    )
                    raise DynamicBlockError(
                        public_message=(
                            "WebSocket connection to Modal endpoint lost after "
                            "the request was sent. The custom Python block may "
                            "have already executed, so the frame was not retried."
                        ),
                        context="modal_executor | websocket_response",
                    )
                if sent_ok:
                    sent_to_container = container_at_send
                last_exc = e
                logger.warning(
                    "[webexec-ws] send/recv failed (attempt %d/%d): %s",
                    attempt + 1,
                    attempts,
                    e,
                )
                continue

        raise DynamicBlockError(
            public_message=f"WebSocket connection to Modal endpoint failed after retry: {last_exc}",
            context="modal_executor | websocket_connection",
        )

    def _recv_bytes_frame(self, ws: Any = None) -> bytes:
        """Receive one frame, requiring it to be binary.

        A text frame here is a protocol violation (typically the payload of
        a server-side close on an already-dead connection); it must be
        treated as connection death, never fed to msgpack.
        """
        resp = (ws if ws is not None else self._ws).recv()
        if not isinstance(resp, bytes):
            raise ConnectionError(
                f"non-binary websocket frame ({type(resp).__name__}); "
                "treating connection as dead"
            )
        return resp

    def _recv_reassembled(self, msgpack: Any) -> bytes:
        """Receive one logical frame, joining chunked frames if signalled."""
        resp_bytes = self._recv_bytes_frame()
        if len(resp_bytes) < 64:
            head = msgpack.unpackb(resp_bytes, raw=False)
            if isinstance(head, dict) and "_chunked" in head:
                return b"".join(
                    self._recv_bytes_frame() for _ in range(head["_chunked"])
                )
        return resp_bytes

    def _execute_ws(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace: str,
        msgpack: Any,
        workflow_context: Dict[str, Any],
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

        request_id = uuid.uuid4().hex
        frame_bytes = self._build_ws_frame(
            python_code=python_code,
            packed_inputs=packed_inputs,
            code_hash=code_hash,
            send_full_code=send_full_code,
            msgpack=msgpack,
            workflow_context=workflow_context,
            request_id=request_id,
        )
        t_pack = _time.monotonic()

        resp_bytes = self._send_recv_with_retry(
            frame_bytes, workspace, request_id=request_id
        )

        t_rtt = _time.monotonic()

        result = msgpack.unpackb(resp_bytes, raw=False)
        self._check_response_id(result, request_id)

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
            # A distinct logical request -> a fresh request_id, so the
            # server's dedup cache can't answer it with the failed response.
            retry_request_id = uuid.uuid4().hex
            retry_frame = self._build_ws_frame(
                python_code=python_code,
                packed_inputs=packed_inputs,
                code_hash=code_hash,
                send_full_code=True,
                msgpack=msgpack,
                workflow_context=workflow_context,
                request_id=retry_request_id,
            )
            resp_bytes = self._send_recv_with_retry(
                retry_frame, workspace, request_id=retry_request_id
            )
            result = msgpack.unpackb(resp_bytes, raw=False)
            self._check_response_id(result, retry_request_id)

        if result.get("success", False):
            self._hashes_sent_on_ws.add(code_hash)
            self._had_success = True

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

        stdout = result.get("stdout")
        stderr = result.get("stderr")
        if stdout:
            sys.stdout.write(stdout)
            sys.stdout.flush()
        if stderr:
            sys.stderr.write(stderr)
            sys.stderr.flush()

        return _deserialize_msgpack_result(result.get("result", {}))

    def _check_response_id(self, result: Any, request_id: str) -> None:
        """Reject a response stamped with a different request's id.

        Only enforced when the server echoes an id (v2); a mismatch means
        the stream carried a stale response, so the connection is dropped.
        """
        if not isinstance(result, dict):
            return
        echoed = result.get("request_id")
        if echoed is not None and echoed != request_id:
            self._drop_ws_connection()
            raise DynamicBlockError(
                public_message=(
                    "WebSocket response did not match the request in flight "
                    "(stale frame on the connection); dropping the connection."
                ),
                context="modal_executor | websocket_response_mismatch",
            )

    @staticmethod
    def _build_ws_frame(
        python_code: PythonCode,
        packed_inputs: Dict[str, Any],
        code_hash: str,
        send_full_code: bool,
        msgpack: Any,
        workflow_context: Dict[str, Any],
        request_id: Optional[str] = None,
    ) -> bytes:
        """Pack a msgpack frame, optionally omitting ``code_str``/``imports``.

        When ``send_full_code`` is False the server resolves the compiled
        namespace through its per-container cache keyed by ``code_hash``.
        """
        payload: Dict[str, Any] = {
            "code_hash": code_hash,
            "run_function_name": python_code.run_function_name,
            "inputs": packed_inputs,
            "workflow_context": workflow_context,
        }
        if request_id is not None:
            payload["request_id"] = request_id
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


class PooledWebSocketModalExecutor:
    """Per-workspace pool of websocket executors.

    A single websocket supports only one in-flight request because responses are
    ordered on the connection. The workspace-level executor cache can therefore
    keep this pool hot without funneling every same-workspace execution through
    one socket.

    Session continuity ("fail loudly when runtime state is lost") is
    per-executor, not per-pool: with a pool size above 1, consecutive frames
    of one stateful run can be routed to different executors — and thus
    different containers with independent runtime state — whenever
    concurrency displaces them from slot 0. Stateful custom Python blocks
    should run with the default pool size of 1.
    """

    def __init__(self, workspace_id: Optional[str] = None):
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        pool_size = max(1, WEBEXEC_WS_CONNECTION_POOL_SIZE)
        self._executors = [
            WebSocketModalExecutor(workspace_id=self.workspace_id)
            for _ in range(pool_size)
        ]
        self._active_counts = [0] * pool_size
        self._pool_lock = threading.Lock()

    def _acquire_executor(self) -> tuple[int, WebSocketModalExecutor]:
        # Prefer the lowest-index least-busy executor so serial workloads
        # (e.g. video streams) reuse a single connection; additional sockets
        # only open when concurrency actually demands them.
        with self._pool_lock:
            best_index = 0
            best_count = self._active_counts[0]
            if best_count > 0:
                for index in range(1, len(self._executors)):
                    active_count = self._active_counts[index]
                    if active_count < best_count:
                        best_index = index
                        best_count = active_count
                        if best_count == 0:
                            break
            self._active_counts[best_index] += 1
            return best_index, self._executors[best_index]

    def _release_executor(self, index: int) -> None:
        with self._pool_lock:
            self._active_counts[index] -= 1

    def close(self) -> None:
        for executor in self._executors:
            executor.close()

    def execute_remote(
        self,
        block_type_name: str,
        python_code: PythonCode,
        inputs: Dict[str, Any],
        workspace_id: Optional[str] = None,
        workflow_context: Optional[Dict[str, Any]] = None,
    ) -> BlockResult:
        index, executor = self._acquire_executor()
        try:
            return executor.execute_remote(
                block_type_name=block_type_name,
                python_code=python_code,
                inputs=inputs,
                workspace_id=workspace_id or self.workspace_id,
                workflow_context=workflow_context,
            )
        finally:
            self._release_executor(index)


def get_modal_executor(workspace_id: Optional[str] = None) -> Any:
    """Returns the right executor based on ``WEBEXEC_TRANSPORT``."""
    from inference.core.env import WEBEXEC_TRANSPORT

    if WEBEXEC_TRANSPORT == "websocket":
        return PooledWebSocketModalExecutor(workspace_id)
    return ModalExecutor(workspace_id)
