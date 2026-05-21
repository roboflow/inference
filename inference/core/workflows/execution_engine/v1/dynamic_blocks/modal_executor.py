"""
Modal executor for Custom Python Blocks in Workflows using Web Endpoints.

This module handles the execution of untrusted user code in Modal sandboxes
using web endpoints for better security and no size limitations.
"""

import base64
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import requests

from inference.core.env import (
    CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS,
    MODAL_ANONYMOUS_WORKSPACE_NAME,
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
)
from inference.core.logger import logger
from inference.core.workflows.errors import (
    DynamicBlockCodeError,
    DynamicBlockError,
    DynamicBlockTimeoutError,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.constants import (
    MODAL_TIMEOUT_ERROR_TYPE,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.entities import (
    PythonCode,
)
from inference.core.workflows.execution_engine.v1.dynamic_blocks.error_utils import (
    build_traceback_string,
    extract_code_snippet,
)
from inference.core.workflows.prototypes.block import BlockResult

# Default and bounds match the Modal handler's watchdog (modal/modal_app.py).
DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS = 20
MIN_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS = 1
MAX_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS = 120
# Client-side buffer added to the watchdog deadline so the server has time
# to send the structured timeout response before the client gives up. See
# design.md "Three-layer timeout" decision.
CLIENT_TIMEOUT_HEADROOM_SECONDS = 10
# Fixed budget used by `validate_code_in_modal` regardless of the
# user-configured per-frame timeout. Validation runs compile()+ast.parse(),
# which complete in milliseconds — keeping this small is defence in depth.
VALIDATION_TIMEOUT_SECONDS = 30

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
            serialized = [patch_for_modal_serialization(item) for item in value]
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

    def __init__(
        self,
        workspace_id: Optional[str] = None,
        custom_python_block_timeout_seconds: Optional[int] = None,
    ):
        """Initialize the Modal executor for a specific workspace.

        Args:
            workspace_id: The workspace ID to namespace execution, defaults to "anonymous".
            custom_python_block_timeout_seconds: Per-frame watchdog deadline in
                seconds. Precedence: constructor arg → env var
                ``CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS`` → 20s default.
                Out-of-range values fall back to the default.
        """
        self.workspace_id = workspace_id or MODAL_ANONYMOUS_WORKSPACE_NAME
        self._base_url = None
        self._timeout_seconds: int = self._resolve_timeout(
            custom_python_block_timeout_seconds
        )

    @staticmethod
    def _resolve_timeout(constructor_arg: Optional[int]) -> int:
        candidate = (
            constructor_arg
            if constructor_arg is not None
            else CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS
        )
        if candidate is None:
            return DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS
        if (
            not isinstance(candidate, int)
            or isinstance(candidate, bool)
            or not (
                MIN_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS
                <= candidate
                <= MAX_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS
            )
        ):
            logger.warning(
                "ModalExecutor: timeout %r is invalid or outside the [%d, %d] range; "
                "falling back to default of %ds.",
                candidate,
                MIN_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS,
                MAX_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS,
                DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS,
            )
            return DEFAULT_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS
        return candidate

    def _get_endpoint_url(self, workspace_id: str) -> str:
        """Get the web endpoint URL for a workspace.

        Args:
            workspace_id: The workspace ID

        Returns:
            The endpoint URL with query parameter for workspace_id
        """
        # Get base URL once (it's the same for all workspace_ids)
        if self._base_url is None:
            # First check for environment variable override
            env_url = os.environ.get("MODAL_WEB_ENDPOINT_URL")
            if env_url:
                self._base_url = env_url
            else:
                # If we couldn't get it dynamically, construct it based on expected pattern
                if not self._base_url:
                    # URL pattern: https://{workspace}--{app}-{class}-{method_truncated}.modal.run
                    # Note: Modal truncates long labels to 63 chars with a hash suffix
                    workspace = MODAL_WORKSPACE_NAME
                    app_name = "webexec"
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
        _timeout_override_seconds: Optional[int] = None,
    ) -> BlockResult:
        """Execute a Custom Python Block in a Modal sandbox via web endpoint.

        Args:
            block_type_name: Name of the block type
            python_code: The Python code to execute
            inputs: Input data for the function
            workspace_id: Optional workspace ID override
            _timeout_override_seconds: Internal override used by
                :func:`validate_code_in_modal` to apply a fixed small budget
                regardless of the configured per-frame timeout. Not part of
                the public API; do not set from external callers.

        Returns:
            BlockResult from the execution

        Raises:
            DynamicBlockError: If Modal credentials are not configured.
            DynamicBlockTimeoutError: If the in-handler watchdog fired or the
                client read timeout fired before the server responded.
            DynamicBlockCodeError: For user-code execution errors and
                non-timeout HTTP/connection failures.
            Exception: If remote execution throws an exception
        """
        # Check if Modal is available
        if not MODAL_AVAILABLE:
            raise DynamicBlockError(
                public_message="Modal credentials not configured. Please set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.",
                context="modal_executor | credentials_check",
            )

        # Resolve effective per-frame timeout (watchdog on the server, plus
        # CLIENT_TIMEOUT_HEADROOM_SECONDS of buffer on the client). The
        # `_timeout_override_seconds` path exists solely for the validation
        # codepath; see VALIDATION_TIMEOUT_SECONDS.
        watchdog_timeout = (
            _timeout_override_seconds
            if _timeout_override_seconds is not None
            else self._timeout_seconds
        )
        request_timeout = watchdog_timeout + CLIENT_TIMEOUT_HEADROOM_SECONDS

        # Use provided workspace_id or fall back to instance default
        workspace = workspace_id if workspace_id else self.workspace_id

        try:
            # Get endpoint URL for this workspace
            endpoint_url = self._get_endpoint_url(workspace)

            # Custom JSON encoder for inputs
            inputs_json = serialize_for_modal_remote_execution(inputs)

            # Prepare request payload
            request_payload = {
                "code_str": python_code.run_function_code,
                "imports": python_code.imports or [],
                "run_function_name": python_code.run_function_name,
                "inputs_json": inputs_json,
                "timeout_seconds": watchdog_timeout,
            }

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

            # Make HTTP request to Modal endpoint
            response = requests.post(
                endpoint_url,
                json=request_payload,
                timeout=request_timeout,
                headers={
                    "Content-Type": "application/json",
                    "Modal-Key": MODAL_TOKEN_ID,
                    "Modal-Secret": MODAL_TOKEN_SECRET,
                },
            )

            # Check HTTP status
            if response.status_code != 200:
                raise DynamicBlockCodeError(
                    public_message=f"Modal endpoint returned status {response.status_code}: {response.text}",
                    context="modal_executor | http_request",
                    block_type_name=block_type_name,
                )

            # Parse response
            result = response.json()

            # Structured timeout response from the in-handler watchdog —
            # handled BEFORE the generic !success branch so the typed
            # exception carries the captured stdout/stderr.
            if result.get("error_type") == MODAL_TIMEOUT_ERROR_TYPE:
                raise DynamicBlockTimeoutError(
                    public_message=(
                        f"Custom Python Block exceeded the configured timeout of "
                        f"{watchdog_timeout}s on this frame. Simplify the block, "
                        "reduce inner loops, or raise the timeout under Advanced "
                        f"Options (max {MAX_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS}s)."
                    ),
                    context="modal_executor | watchdog_timeout",
                    block_type_name=block_type_name,
                    stdout=result.get("stdout"),
                    stderr=result.get("stderr"),
                )

            # Check for errors
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

                # Propagate DynamicBlockCodeError on runtime error. Will pass through
                # the core executor and be handled by its own HTTP handler.
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

        except requests.exceptions.ReadTimeout as e:
            # Defence in depth: the server-side watchdog should respond first,
            # but if it didn't, surface the same typed exception so callers
            # don't have to special-case the network path.
            raise DynamicBlockTimeoutError(
                public_message=(
                    f"Custom Python Block exceeded the configured timeout of "
                    f"{watchdog_timeout}s on this frame (client read timeout). "
                    "Simplify the block, reduce inner loops, or raise the timeout "
                    f"under Advanced Options (max {MAX_CUSTOM_PYTHON_BLOCK_TIMEOUT_SECONDS}s)."
                ),
                context="modal_executor | client_read_timeout",
                block_type_name=block_type_name,
            ) from e
        except requests.exceptions.RequestException as e:
            # Non-timeout HTTP/connection failure. Re-class to the execution-
            # engine side (was previously DynamicBlockError, a compiler-side
            # exception, which categorised runtime failures incorrectly).
            raise DynamicBlockCodeError(
                public_message=f"Failed to connect to Modal endpoint: {str(e)}",
                context="modal_executor | http_connection",
                block_type_name=block_type_name,
            ) from e


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
        # For validation, we don't need complex inputs, just pass empty JSON.
        # `_timeout_override_seconds=VALIDATION_TIMEOUT_SECONDS` pins the
        # validation budget to a small fixed value (compile + ast.parse run
        # in milliseconds) regardless of the user-configured per-frame
        # timeout — defence in depth, see design.md decision 8.
        result = executor.execute_remote(
            block_type_name="validation",
            python_code=validation_code,
            inputs={},
            workspace_id=workspace,
            _timeout_override_seconds=VALIDATION_TIMEOUT_SECONDS,
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
