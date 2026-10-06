"""Importing this module installs the process-wide `WorkflowsConfiguration`, which must
happen before anything imports `roboflow_workflows.environment`."""

from __future__ import annotations

import functools
import importlib
import json
import logging
import os
import platform
import re
import socket
import stat
import threading
import time
import urllib.parse
import uuid
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import requests
from roboflow_workflows.configuration import (
    DebugConfiguration,
    EngineConfiguration,
    FontsConfiguration,
    ModalConfiguration,
    ModelsConfiguration,
    PlatformConfiguration,
    RemoteExecutionConfiguration,
    SecretsConfiguration,
    TensorConfiguration,
    WorkflowsConfiguration,
    configure_process,
    resolve_image_tensor_device,
)

from inference_models.utils.environment import (
    get_boolean_from_env,
    get_float_from_env,
    get_integer_from_env,
)
from inference_server import configuration

logger = logging.getLogger(__name__)

_ALLOWED_API_KEY_TRANSPORTS = ("legacy", "both", "header")
_API_PROXY_ENDPOINT_PREFIXES = ("apiproxy", "api-proxy")
_API_KEY_PATTERN = re.compile(r"api_key=(.[^&]*)")
_MIN_KEY_LENGTH_TO_REVEAL_PREFIX = 8


def _csv(raw: str) -> Tuple[str, ...]:
    return tuple(entry.strip().lower() for entry in raw.split(",") if entry.strip())


def _optional_csv(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
    if raw is None:
        return None
    return tuple(entry.strip() for entry in raw.split(",") if entry.strip())


def _optional_address_set(raw: Optional[str]) -> Optional[Tuple[str, ...]]:
    if raw is None:
        return None
    return tuple(sorted(set(raw.split(","))))


def build_workflows_configuration() -> WorkflowsConfiguration:
    offline_mode = configuration.LEGACY_OFFLINE_MODE
    secure_gateway = configuration.SECURE_GATEWAY
    step_execution_mode = os.environ.get(
        "WORKFLOWS_STEP_EXECUTION_MODE", "local"
    ).lower()
    remote_api_target = os.environ.get("WORKFLOWS_REMOTE_API_TARGET", "hosted").lower()
    remote_api_key_transport = os.environ.get(
        "WORKFLOWS_REMOTE_API_KEY_TRANSPORT", "both"
    ).lower()
    if remote_api_key_transport not in _ALLOWED_API_KEY_TRANSPORTS:
        raise ValueError(
            f"Invalid WORKFLOWS_REMOTE_API_KEY_TRANSPORT: "
            f"{remote_api_key_transport!r}. Expected one of: "
            f"{list(_ALLOWED_API_KEY_TRANSPORTS)}."
        )
    if offline_mode and step_execution_mode == "remote":
        warnings.warn(
            "WORKFLOWS_STEP_EXECUTION_MODE=remote is not available while OFFLINE_MODE "
            "is enabled. Forcing local workflow step execution.",
            stacklevel=1,
        )
        step_execution_mode = "local"
    elif (
        secure_gateway
        and step_execution_mode == "remote"
        and remote_api_target == "hosted"
    ):
        warnings.warn(
            "WORKFLOWS_STEP_EXECUTION_MODE=remote with WORKFLOWS_REMOTE_API_TARGET=hosted "
            "is not supported behind SECURE_GATEWAY - hosted Roboflow inference endpoints "
            "are not reachable through the gateway proxy. Forcing local step execution. "
            "Use WORKFLOWS_REMOTE_API_TARGET=self-hosted (with LOCAL_INFERENCE_API_URL "
            "pointing at a server inside the gateway perimeter) to keep remote execution.",
            stacklevel=1,
        )
        step_execution_mode = "local"
    custom_python_execution_mode = (
        os.environ.get("WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE", "local")
        .strip()
        .lower()
    )
    if custom_python_execution_mode not in {"local", "modal"}:
        raise ValueError(
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE must be local or modal"
        )
    if offline_mode and custom_python_execution_mode == "modal":
        raise RuntimeError(
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE=modal cannot run in OFFLINE_MODE. "
            "Disable offline mode to retain sandbox isolation, or explicitly configure "
            "local execution only for trusted custom Python workflows."
        )
    sam3_exec_mode = os.environ.get("SAM3_EXEC_MODE", "local").lower()
    if sam3_exec_mode not in {"local", "remote"}:
        raise ValueError(
            f"Invalid SAM3 execution mode in ENVIRONMENT var SAM3_EXEC_MODE "
            f"(local or remote): {sam3_exec_mode}"
        )
    if offline_mode and sam3_exec_mode == "remote":
        warnings.warn(
            "SAM3_EXEC_MODE=remote is not available while OFFLINE_MODE is enabled. "
            "Forcing local SAM3 execution.",
            stacklevel=1,
        )
        sam3_exec_mode = "local"
    project = os.environ.get("PROJECT", "roboflow-platform")
    hosted_platform = project == "roboflow-platform"
    tensor_representation_enabled = get_boolean_from_env(
        "ENABLE_TENSOR_DATA_REPRESENTATION", default=False
    )
    sam_video_mask_representation = (
        os.environ.get("WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION", "rle").strip().lower()
    )
    if sam_video_mask_representation not in {"rle", "dense"}:
        warnings.warn(
            "Invalid value of `WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION` variable: "
            f"{sam_video_mask_representation!r} - allowed values are 'rle' "
            "and 'dense'. Falling back to 'rle'.",
            stacklevel=1,
        )
        sam_video_mask_representation = "rle"
    modal_token_id = os.environ.get("MODAL_TOKEN_ID")
    modal_token_secret = os.environ.get("MODAL_TOKEN_SECRET")
    return WorkflowsConfiguration(
        engine=EngineConfiguration(
            step_execution_mode=step_execution_mode,
            async_future_result_timeout=get_float_from_env(
                "WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT", default=60.0
            ),
            max_inner_workflow_depth=get_integer_from_env(
                "WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH", default=4
            ),
            max_inner_workflow_count=get_integer_from_env(
                "WORKFLOWS_MAX_INNER_WORKFLOW_COUNT", default=32
            ),
            allow_custom_python_execution=get_boolean_from_env(
                "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS", default=True
            ),
            custom_python_execution_mode=custom_python_execution_mode,
            allow_blocks_accessing_local_storage=get_boolean_from_env(
                "ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE", default=True
            ),
            allow_blocks_accessing_environmental_variables=get_boolean_from_env(
                "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES", default=True
            ),
            blocks_write_directory=os.environ.get("WORKFLOW_BLOCKS_WRITE_DIRECTORY"),
            disabled_block_types=_csv(
                os.environ.get("WORKFLOW_DISABLED_BLOCK_TYPES", "")
            ),
            disabled_block_patterns=_csv(
                os.environ.get("WORKFLOW_DISABLED_BLOCK_PATTERNS", "")
            ),
            allow_webhook_sink_to_non_global_addresses=get_boolean_from_env(
                "ALLOW_WEBHOOK_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", default=True
            ),
            allow_postgresql_sink_to_non_global_addresses=get_boolean_from_env(
                "ALLOW_POSTGRESQL_WORKFLOWS_SINK_TO_NON_GLOBAL_ADDRESSES", default=True
            ),
            postgresql_sink_blacklisted_addresses=_optional_address_set(
                os.environ.get("POSTGRESQL_WORKFLOWS_SINK_BLACKLISTED_ADDRESSES")
            ),
            postgresql_sink_whitelisted_addresses=_optional_address_set(
                os.environ.get("POSTGRESQL_WORKFLOWS_SINK_WHITELISTED_ADDRESSES")
            ),
            allow_kafka_sinks_user_provided_bootstrap_servers=get_boolean_from_env(
                "KAFKA_WORKFLOWS_SINKS_ALLOW_USER_PROVIDED_BOOTSTRAP_SERVERS",
                default=True,
            ),
            kafka_sinks_whitelisted_bootstrap_servers=_optional_csv(
                os.environ.get("KAFKA_WORKFLOWS_SINKS_WHITELISTED_BOOTSTRAP_SERVERS")
            ),
            allow_mqtt_blocks_user_provided_host=configuration.MQTT_WORKFLOWS_BLOCKS_ALLOW_USER_PROVIDED_HOST,
            mqtt_blocks_whitelisted_hosts=configuration.MQTT_WORKFLOWS_BLOCKS_WHITELISTED_HOSTS,
        ),
        tensor=TensorConfiguration(
            representation_enabled=tensor_representation_enabled,
            image_tensor_device=resolve_image_tensor_device(
                representation_enabled=tensor_representation_enabled,
                device=os.environ.get("WORKFLOWS_IMAGE_TENSOR_DEVICE"),
            ),
            visualisation_validate_owners=get_boolean_from_env(
                "WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS", default=False
            ),
            sam_video_mask_representation=sam_video_mask_representation,
            enforce_dense_instance_masks=get_boolean_from_env(
                "WORKFLOWS_ENFORCE_DENSE_INSTANCE_MASKS", default=False
            ),
        ),
        remote=RemoteExecutionConfiguration(
            api_target=remote_api_target,
            api_key_transport=remote_api_key_transport,
            local_inference_api_url=os.environ.get(
                "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001"
            ),
            hosted_detect_url=os.environ.get(
                "HOSTED_DETECT_URL",
                (
                    "https://detect.roboflow.com"
                    if hosted_platform
                    else "https://lambda-object-detection.staging.roboflow.com"
                ),
            ),
            hosted_classification_url=os.environ.get(
                "HOSTED_CLASSIFICATION_URL",
                (
                    "https://classify.roboflow.com"
                    if hosted_platform
                    else "https://lambda-classification.staging.roboflow.com"
                ),
            ),
            hosted_instance_segmentation_url=os.environ.get(
                "HOSTED_INSTANCE_SEGMENTATION_URL",
                (
                    "https://outline.roboflow.com"
                    if hosted_platform
                    else "https://lambda-instance-segmentation.staging.roboflow.com"
                ),
            ),
            hosted_semantic_segmentation_url=os.environ.get(
                "HOSTED_SEMANTIC_SEGMENTATION_URL",
                (
                    "https://segment.roboflow.com"
                    if hosted_platform
                    else "https://lambda-semantic-segmentation.staging.roboflow.com"
                ),
            ),
            hosted_core_model_url=os.environ.get(
                "HOSTED_CORE_MODEL_URL",
                (
                    "https://infer.roboflow.com"
                    if hosted_platform
                    else "https://3hkaykeh3j.execute-api.us-east-1.amazonaws.com"
                ),
            ),
            max_step_batch_size=get_integer_from_env(
                "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", default=1
            ),
            max_step_concurrent_requests=get_integer_from_env(
                "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", default=8
            ),
            inner_workflow_remote_target=os.environ.get(
                "WORKFLOWS_INNER_WORKFLOW_REMOTE_TARGET",
                "https://serverless.roboflow.com",
            ),
            inner_workflow_remote_dispatch_request_timeout=get_float_from_env(
                "WORKFLOWS_INNER_WORKFLOW_REMOTE_DISPATCH_REQUEST_TIMEOUT",
                default=300.0,
            ),
            openai_compatible_allowed_base_urls=tuple(
                sorted(
                    {
                        url.strip().rstrip("/")
                        for url in os.environ.get(
                            "OPENAI_COMPATIBLE_ALLOWED_BASE_URLS", "*"
                        ).split(",")
                    }
                )
            ),
        ),
        platform=PlatformConfiguration(
            api_base_url=configuration.API_BASE_URL,
            offline_mode=offline_mode,
            secure_gateway=secure_gateway,
            gcp_serverless=configuration.GCP_SERVERLESS,
            lambda_runtime=configuration.LAMBDA,
        ),
        fonts=FontsConfiguration(
            allow_download=configuration.ALLOW_WORKFLOWS_FONTS_DOWNLOAD,
            model_cache_dir=configuration.MODEL_CACHE_DIR,
        ),
        models=ModelsConfiguration(
            lmm_enabled=get_boolean_from_env("LMM_ENABLED", default=False),
            vlm_segmentation_max_polygon_vertices=get_integer_from_env(
                "WORKFLOWS_VLM_SEGMENTATION_MAX_POLYGON_VERTICES", default=500
            ),
            clip_version_id=os.environ.get("CLIP_VERSION_ID", "ViT-B-16"),
            core_model_sam2_enabled=get_boolean_from_env(
                "CORE_MODEL_SAM2_ENABLED", default=True
            ),
            core_model_sam3_enabled=get_boolean_from_env(
                "CORE_MODEL_SAM3_ENABLED", default=True
            ),
            core_model_pe_enabled=get_boolean_from_env(
                "CORE_MODEL_PE_ENABLED", default=True
            ),
            core_model_gaze_enabled=get_boolean_from_env(
                "CORE_MODEL_GAZE_ENABLED", default=True
            ),
            sam3_exec_mode=sam3_exec_mode,
            sam3_3d_objects_enabled=get_boolean_from_env(
                "SAM3_3D_OBJECTS_ENABLED", default=False
            ),
            florence2_enabled=get_boolean_from_env("FLORENCE2_ENABLED", default=True),
            qwen_2_5_enabled=get_boolean_from_env("QWEN_2_5_ENABLED", default=True),
            qwen_3_enabled=get_boolean_from_env("QWEN_3_ENABLED", default=True),
            qwen_3_5_enabled=get_boolean_from_env("QWEN_3_5_ENABLED", default=True),
            smolvlm2_enabled=get_boolean_from_env("SMOLVLM2_ENABLED", default=True),
            moondream2_enabled=get_boolean_from_env("MOONDREAM2_ENABLED", default=True),
            depth_estimation_enabled=get_boolean_from_env(
                "DEPTH_ESTIMATION_ENABLED", default=True
            ),
            cosmos3_enabled=get_boolean_from_env("COSMOS3_ENABLED", default=True),
            glm_ocr_enabled=get_boolean_from_env("GLM_OCR_ENABLED", default=True),
        ),
        modal=ModalConfiguration(
            token_id=modal_token_id.strip("\"'") if modal_token_id else None,
            token_secret=(
                modal_token_secret.strip("\"'") if modal_token_secret else None
            ),
            workspace_name=os.environ.get("MODAL_WORKSPACE_NAME", "roboflow"),
            allow_anonymous_execution=get_boolean_from_env(
                "MODAL_ALLOW_ANONYMOUS_EXECUTION", default=False
            ),
            anonymous_workspace_name=os.environ.get(
                "MODAL_ANONYMOUS_WORKSPACE_NAME", "anonymous"
            ),
            app_name=os.environ.get("WEBEXEC_MODAL_APP_NAME", f"webexec-{project}"),
            executor_idle_ttl_seconds=get_integer_from_env(
                "WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS", default=1800
            ),
            jpeg_quality=get_integer_from_env("WEBEXEC_JPEG_QUALITY", default=95),
            transport=os.environ.get("WEBEXEC_TRANSPORT", "http").lower().strip(),
            ws_connect_timeout_seconds=get_integer_from_env(
                "WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS", default=30
            ),
            ws_read_timeout_seconds=get_integer_from_env(
                "WEBEXEC_WS_READ_TIMEOUT_SECONDS", default=720
            ),
            ws_connection_pool_size=get_integer_from_env(
                "WEBEXEC_WS_CONNECTION_POOL_SIZE", default=1
            ),
            ws_fail_on_session_loss=get_boolean_from_env(
                "WEBEXEC_WS_FAIL_ON_SESSION_LOSS", default=False
            ),
            ws_idle_release_seconds=get_integer_from_env(
                "WEBEXEC_WS_IDLE_RELEASE_SECONDS", default=120
            ),
        ),
        secrets=SecretsConfiguration(
            api_key=os.environ.get("ROBOFLOW_API_KEY") or os.environ.get("API_KEY"),
            roboflow_internal_service_name=configuration.ROBOFLOW_INTERNAL_SERVICE_NAME,
            roboflow_internal_service_secret=configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET,
        ),
        debug=DebugConfiguration(
            output_dir=os.environ.get("INFERENCE_DEBUG_OUTPUT_DIR"),
        ),
    )


ENTERPRISE_BLOCKS_PLUGIN = "roboflow_workflows.enterprise_blocks.loader"


def expand_enterprise_blocks_plugin() -> None:
    """Prepend the enterprise blocks loader to `WORKFLOWS_PLUGINS` when enabled.

    The loader goes first so the block order stays core, enterprise, then custom
    plugins. A loader that is already listed keeps its position.
    """
    if not configuration.LOAD_ENTERPRISE_BLOCKS:
        return

    plugins = [
        plugin
        for plugin in os.environ.get("WORKFLOWS_PLUGINS", "").split(",")
        if plugin
    ]
    if ENTERPRISE_BLOCKS_PLUGIN in plugins:
        return

    os.environ["WORKFLOWS_PLUGINS"] = ",".join([ENTERPRISE_BLOCKS_PLUGIN] + plugins)


def require_enterprise_blocks_plugin() -> None:
    """Fail at startup when enterprise blocks are enabled but cannot be imported.

    Raises:
        RuntimeError: When `LOAD_ENTERPRISE_BLOCKS` is set and the enterprise
            loader or one of its dependencies is missing.
    """
    if not configuration.LOAD_ENTERPRISE_BLOCKS:
        return

    try:
        importlib.import_module(ENTERPRISE_BLOCKS_PLUGIN)
    except ImportError as error:
        raise RuntimeError(
            "LOAD_ENTERPRISE_BLOCKS is enabled but the enterprise Workflow blocks "
            f"cannot be imported ({error}). Install the `enterprise` extra: "
            "roboflow-workflows[enterprise]."
        ) from error


expand_enterprise_blocks_plugin()
SERVER_WORKFLOWS_CONFIGURATION = build_workflows_configuration()
configure_process(SERVER_WORKFLOWS_CONFIGURATION)
require_enterprise_blocks_plugin()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from roboflow_workflows.errors import (  # noqa: E402
    ClientCausedStepExecutionError,
    RuntimeLimitsCausedStepExecutionError,
    WorkflowDefinitionError,
    WorkflowImageLoadError,
)
from roboflow_workflows.prototypes.image_codec import (  # noqa: E402
    WorkflowsLocalImageCodec,
    set_image_codec,
)
from roboflow_workflows.prototypes.platform_client import (  # noqa: E402
    HttpErrorHandlers,
)
from roboflow_workflows.prototypes.platform_errors import (  # noqa: E402
    FeatureDeprecatedError,
    RoboflowAPIConnectionError,
    RoboflowAPIForbiddenError,
    RoboflowAPINotAuthorizedError,
    RoboflowAPINotNotFoundError,
    RoboflowAPIRequestError,
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
from roboflow_workflows.utils.image_encoding import (  # noqa: E402
    choose_image_decoding_flags,
    convert_gray_image_to_bgr,
    decode_encoded_image_bytes,
)
from inference_model_manager.pipelines import InvalidPipelineIdError  # noqa: E402
from inference_models.errors import (  # noqa: E402
    ModelNotFoundError,
    ModelPackageAlternativesExhaustedError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    UnauthorizedModelAccessError,
)
from inference_sdk.http.errors import HTTPCallErrorError  # noqa: E402
from inference_server import platform_http, telemetry  # noqa: E402
from inference_server.errors import ServerBusyError  # noqa: E402
from inference_server.framework.input_parsers.url_fetch import (  # noqa: E402
    URL_FETCH_TIMEOUT_S,
)
from inference_server.framework.model_stat import _TtlLruCache  # noqa: E402
from inference_server.legacy import bridge as legacy_bridge  # noqa: E402
from inference_server.legacy.bridge import LoopBridge  # noqa: E402
from inference_server.legacy.errors import (  # noqa: E402
    MODEL_ACCESS_ERROR_MESSAGES,
    NOT_FOUND_MESSAGE,
    REGISTRY_REQUEST_FAILED_MESSAGE,
    UNAUTHORIZED_MESSAGE,
    ImageFetchError,
    LegacyHTTPError,
    ModelNotReadyError,
)
from inference_server.legacy.telemetry_recording import record_telemetry  # noqa: E402
from inference_server.platform_http import (  # noqa: E402
    API_REQUEST_TIMEOUT_S,
    _add_params_to_url,
    _platform_request,
)
from inference_server.workflows import definition_cache  # noqa: E402
from inference_server.workflows.errors import (  # noqa: E402
    MalformedRoboflowAPIResponseError,
    ModelDeploymentNotSupportedError,
    PaymentRequiredError,
    RoboflowAPIUsagePausedError,
    WorkspaceLoadError,
)
from inference_server.workflows.redis_cache import build_workflows_cache  # noqa: E402

_URL_FETCH_BRIDGE_TIMEOUT_S = URL_FETCH_TIMEOUT_S + 5
_IMAGE_LOADING_CONTEXT = "workflow_execution | image_loading"
_NUMPY_INPUT_REFUSAL = (
    "NumPy image type is not supported in this configuration of `inference`."
)
_STEP_EXECUTION_CONTEXT = "workflow_execution | step_execution"
_MODEL_ACCESS_ERROR_MESSAGES = {
    402: "Not enough credits to execute step {step_name}. Verify your workspace billing page.",
    403: "Forbidden error occurred while execution of step {step_name}. "
    "This error usually means there is a problem with the Roboflow API key.",
    423: "Roboflow API usage is paused while executing step {step_name}. "
    "Contact your workspace administrator to re-enable API keys.",
}


def _redact_api_key(value: str) -> str:
    def _replace(match: re.Match) -> str:
        key = match.group(1)
        if len(key) < _MIN_KEY_LENGTH_TO_REVEAL_PREFIX:
            return "api_key=***"
        return f"api_key={key[:2]}***{key[-2:]}"

    return _API_KEY_PATTERN.sub(_replace, value)


def _add_params_to_url(url: str, params: List[Tuple[str, str]]) -> str:
    if not params:
        return url
    query = "&".join(
        f"{name}={urllib.parse.quote_plus(value)}" for name, value in params
    )
    return f"{url}?{query}"


def _is_successful(response: requests.Response) -> bool:
    return 200 <= response.status_code < 300


def _api_error_message(response: requests.Response, api_key: Optional[str]) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    message = None
    if isinstance(payload, dict):
        message = payload.get("message") or payload.get("error")
    if not isinstance(message, str):
        message = f"Roboflow API request failed with status {response.status_code}"
    message = _redact_api_key(message)
    if api_key:
        message = message.replace(api_key, "***")
    return message


_SERVICE_SECRET_PATTERN = re.compile(r"service_secret=[^&]*")
_WORKSPACE_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
# Process-wide api_key -> workspace cache, the counterpart of the legacy
# server's `@ttl_cache` on `get_roboflow_workspace`: blocks such as
# visual_search_classifier look the workspace up once per image. Keyed by the
# key's SHA-256 so no plaintext key is held; only successful lookups are stored.
_WORKSPACE_CACHE = _TtlLruCache(
    configuration.WORKSPACE_CACHE_MAX_SIZE, configuration.WORKSPACE_CACHE_TTL_S
)
_WORKSPACE_CACHE_LOCK = threading.Lock()


def clear_workspace_cache() -> None:
    with _WORKSPACE_CACHE_LOCK:
        _WORKSPACE_CACHE.clear()


_SEARCH_DEFAULT_FIELDS = [
    "id",
    "name",
    "filename",
    "url",
    "user_metadata",
    "tags",
    "width",
    "height",
    "aspectRatio",
]
_NOT_AUTHORIZED_MESSAGE = (
    "Unauthorized access to roboflow API - check API key. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
_FORBIDDEN_MESSAGE = (
    "Unauthorized access to roboflow API - check API key regarding correctness and "
    "required scopes. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
# Same classes and messages as `inference.core.roboflow_api.wrap_roboflow_api_errors`.
_PLATFORM_API_ERRORS: Dict[int, Tuple[type, str]] = {
    401: (RoboflowAPINotAuthorizedError, _NOT_AUTHORIZED_MESSAGE),
    402: (
        PaymentRequiredError,
        "Not enough credits to perform this request. Verify your workspace billing page.",
    ),
    403: (RoboflowAPIForbiddenError, _FORBIDDEN_MESSAGE),
    404: (
        RoboflowAPINotNotFoundError,
        "Could not find requested Roboflow resource. Check that the provided dataset "
        "and version are correct, and check that the provided Roboflow API key has "
        "the correct permissions.",
    ),
    423: (
        RoboflowAPIUsagePausedError,
        "Roboflow API usage is paused. Please contact your workspace administrator "
        "to re-enable api keys.",
    ),
}


_WORKFLOW_FETCH_FAILURE_MESSAGES = {
    401: UNAUTHORIZED_MESSAGE,
    404: NOT_FOUND_MESSAGE,
    **MODEL_ACCESS_ERROR_MESSAGES,
}


def _api_key_safe_raise_for_status(response: requests.Response) -> None:
    if response.status_code < 400:
        return None
    response.url = _SERVICE_SECRET_PATTERN.sub(
        "service_secret=***", _redact_api_key(response.url)
    )
    response.raise_for_status()


def _translate_platform_api_errors(
    call: Callable[[], Any],
    http_error_overrides: Optional[Dict[int, Tuple[type, str]]] = None,
) -> Any:
    """Map transport failures onto the shared `RoboflowAPI*` error classes, as
    the `inference` server does for the Roboflow-platform blocks."""
    try:
        return call()
    except requests.exceptions.Timeout as error:
        raise RoboflowAPITimeoutError(
            "Timeout when attempting to connect to Roboflow API."
        ) from error
    except (requests.exceptions.ConnectionError, ConnectionError) as error:
        raise RoboflowAPIConnectionError(
            "Could not connect to Roboflow API."
        ) from error
    except requests.exceptions.HTTPError as error:
        status_code = error.response.status_code
        handlers = {**_PLATFORM_API_ERRORS, **(http_error_overrides or {})}
        if status_code in handlers:
            error_class, message = handlers[status_code]
            raise error_class(message) from error
        raise _unsuccessful_request_error(status_code) from error
    except (requests.exceptions.InvalidJSONError, ValueError) as error:
        raise MalformedRoboflowAPIResponseError(
            "Could not decode JSON response from Roboflow API."
        ) from error


def _platform_api_error(status_code: int) -> Exception:
    if status_code in _PLATFORM_API_ERRORS:
        error_class, message = _PLATFORM_API_ERRORS[status_code]
        return error_class(message)

    return _unsuccessful_request_error(status_code)


def _unsuccessful_request_error(
    status_code: int,
) -> RoboflowAPIUnsuccessfulRequestError:
    return RoboflowAPIUnsuccessfulRequestError(
        f"Unsuccessful request to Roboflow API with response code: {status_code}"
    )


def _refuse_when_offline(operation: str) -> None:
    if configuration.LEGACY_OFFLINE_MODE:
        raise RoboflowAPIConnectionError(
            f"Cannot {operation} at Roboflow - OFFLINE_MODE is enabled."
        )


def _records_api_call(function_name: str) -> Callable:
    def decorator(function: Callable) -> Callable:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            finally:
                record_telemetry(
                    telemetry.record_api_call,
                    function_name,
                    time.perf_counter() - started,
                )

            return result

        return wrapper

    return decorator


def _api_base_url_for_endpoint(endpoint: str) -> str:
    """Return the platform base URL serving ``endpoint``, proxy prefixes included."""
    normalized_endpoint = endpoint.strip("/")
    for prefix in _API_PROXY_ENDPOINT_PREFIXES:
        if normalized_endpoint == prefix or normalized_endpoint.startswith(
            f"{prefix}/"
        ):
            return configuration.API_PROXY_BASE_URL

    return configuration.API_BASE_URL


def _api_url(path: str) -> str:
    return f"{configuration.API_BASE_URL.rstrip('/')}/{path}"


def collect_system_info() -> dict:
    """Platform, architecture, hostname, IP, MAC and processor, best effort.

    Same keys as `inference.core.managers.metrics.get_system_info`.
    """
    info = {}
    try:
        info["platform"] = platform.system()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip_address"] = socket.gethostbyname(socket.gethostname())
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
    except Exception as error:
        logger.exception(error)
    return info


class ServerRoboflowPlatformClient:
    @_records_api_call("_make_request")
    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        _refuse_when_offline(operation="make API requests")
        url_params: List[Tuple[str, str]] = []
        if api_key:
            url_params.append(("api_key", api_key))
        if params:
            url_params.extend(params)
        base_url = _api_base_url_for_endpoint(endpoint)
        url = _add_params_to_url(
            url=f"{base_url.rstrip('/')}/{endpoint.strip('/')}",
            params=url_params,
        )
        try:
            response = _platform_request(
                "post",
                self.wrap_url(url),
                json=payload,
                headers=self.build_api_headers(),
                timeout=API_REQUEST_TIMEOUT_S,
            )
        except LegacyHTTPError as error:
            if error.status_code == 504:
                raise RoboflowAPITimeoutError(
                    "Timeout when attempting to connect to Roboflow API."
                ) from None
            raise RoboflowAPIConnectionError(
                "Could not connect to Roboflow API."
            ) from None
        if not _is_successful(response):
            handler = (http_errors_handlers or {}).get(response.status_code)
            if handler is None:
                raise _platform_api_error(response.status_code)

            message = _api_error_message(response, api_key)
            handler(requests.exceptions.HTTPError(message, response=response))
            raise _unsuccessful_request_error(response.status_code)
        return response.json()

    def build_api_headers(
        self, explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        return platform_http.build_api_headers(explicit_headers=explicit_headers)

    def build_weights_provider_headers(
        self,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        if configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET:
            return self.build_api_headers(
                explicit_headers={
                    "X-Roboflow-Internal-Service-Secret": configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET
                }
            )
        return self.build_api_headers()

    def wrap_url(self, url: str) -> str:
        return platform_http.wrap_url(url)

    # The Roboflow-platform blocks' operations. Endpoints, payloads, query
    # parameters and error classes are those of `inference.core.roboflow_api`.

    def _post_to_api(
        self, url: str, headers: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> requests.Response:
        response = requests.post(
            url=self.wrap_url(url),
            headers=headers if headers is not None else self.build_api_headers(),
            timeout=API_REQUEST_TIMEOUT_S,
            **platform_http.tls_verification_options(),
            **kwargs,
        )
        _api_key_safe_raise_for_status(response=response)
        return response

    def get_roboflow_workspace(self, api_key: str) -> str:
        if not api_key:
            raise WorkspaceLoadError("Empty workspace encountered, check your API key.")
        cache_key = sha256(api_key.encode("utf-8")).hexdigest()
        with _WORKSPACE_CACHE_LOCK:
            cached_workspace_id = _WORKSPACE_CACHE.get(cache_key)
        if cached_workspace_id is not None:
            return cached_workspace_id
        workspace_id = self._fetch_roboflow_workspace(api_key=api_key)
        with _WORKSPACE_CACHE_LOCK:
            _WORKSPACE_CACHE.set(cache_key, workspace_id)
        return workspace_id

    @_records_api_call("get_roboflow_workspace")
    def _fetch_roboflow_workspace(self, api_key: str) -> str:
        # Guarded behind the cache lookup: the legacy server's `ttl_cache`
        # still answers for an already-resolved key while OFFLINE_MODE is on.
        _refuse_when_offline(operation="fetch workspace")
        url = _add_params_to_url(
            url=_api_url(""), params=[("api_key", api_key), ("nocache", "true")]
        )

        def _call() -> dict:
            response = requests.get(
                url=self.wrap_url(url),
                headers=self.build_api_headers(),
                timeout=API_REQUEST_TIMEOUT_S,
                **platform_http.tls_verification_options(),
            )
            _api_key_safe_raise_for_status(response=response)
            return response.json()

        workspace_id = _translate_platform_api_errors(_call).get("workspace")
        if not isinstance(workspace_id, str) or not _WORKSPACE_ID_PATTERN.fullmatch(
            workspace_id
        ):
            raise WorkspaceLoadError("Empty workspace encountered, check your API key.")
        return workspace_id

    def _get_from_api(self, url: str) -> dict:
        def _call() -> dict:
            response = requests.get(
                url=self.wrap_url(url),
                headers=self.build_api_headers(),
                timeout=API_REQUEST_TIMEOUT_S,
                **platform_http.tls_verification_options(),
            )
            _api_key_safe_raise_for_status(response=response)
            return response.json()

        parsed_response = _translate_platform_api_errors(_call)

        return parsed_response

    @_records_api_call("get_roboflow_dataset_type")
    def get_roboflow_dataset_type(
        self, api_key: str, workspace_id: str, dataset_id: str
    ) -> str:
        """Fetch the task type of a Roboflow project.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace owning the project.
            dataset_id: Project identifier.

        Returns:
            The project type, ``object-detection`` when the platform reports none.

        Raises:
            RoboflowAPIRequestError: If the platform cannot be reached or rejects
                the request.
        """
        _refuse_when_offline(operation="fetch dataset type")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/{dataset_id}"),
            params=[("api_key", api_key), ("nocache", "true")],
        )

        project = self._get_from_api(url).get("project", {})
        if "type" not in project:
            logger.warning(
                "Project task type not defined for workspace=%s and dataset=%s, "
                "defaulting to object-detection.",
                workspace_id,
                dataset_id,
            )
        dataset_type = project.get("type", "object-detection")

        return dataset_type

    @_records_api_call("get_roboflow_active_learning_configuration")
    def get_roboflow_active_learning_configuration(
        self, api_key: str, workspace_id: str, dataset_id: str
    ) -> dict:
        """Fetch the active learning configuration of a Roboflow project.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace owning the project.
            dataset_id: Project identifier.

        Returns:
            The configuration document as the platform returns it.

        Raises:
            RoboflowAPIRequestError: If the platform cannot be reached or rejects
                the request.
        """
        _refuse_when_offline(operation="fetch active learning configuration")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/{dataset_id}/active_learning"),
            params=[("api_key", api_key)],
        )

        active_learning_configuration = self._get_from_api(url)

        return active_learning_configuration

    @_records_api_call("get_roboflow_labeling_batches")
    def get_roboflow_labeling_batches(
        self, api_key: str, workspace_id: str, dataset_id: str
    ) -> dict:
        """Fetch the labeling batches of a Roboflow project.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace owning the project.
            dataset_id: Project identifier.

        Returns:
            The platform response, with the batches under ``batches``.

        Raises:
            RoboflowAPIRequestError: If the platform cannot be reached or rejects
                the request.
        """
        _refuse_when_offline(operation="fetch labeling batches")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/{dataset_id}/batches"),
            params=[("api_key", api_key)],
        )

        labeling_batches = self._get_from_api(url)

        return labeling_batches

    @_records_api_call("get_roboflow_labeling_jobs")
    def get_roboflow_labeling_jobs(
        self, api_key: str, workspace_id: str, dataset_id: str
    ) -> dict:
        """Fetch the labeling jobs of a Roboflow project.

        Args:
            api_key: Roboflow API key.
            workspace_id: Workspace owning the project.
            dataset_id: Project identifier.

        Returns:
            The platform response, with the jobs under ``jobs``.

        Raises:
            RoboflowAPIRequestError: If the platform cannot be reached or rejects
                the request.
        """
        _refuse_when_offline(operation="fetch labeling jobs")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/{dataset_id}/jobs"),
            params=[("api_key", api_key)],
        )

        labeling_jobs = self._get_from_api(url)

        return labeling_jobs

    @_records_api_call("add_custom_metadata")
    def add_custom_metadata(
        self,
        api_key: str,
        workspace_id: str,
        inference_ids: List[str],
        field_name: str,
        field_value: str,
    ) -> None:
        if configuration.LEGACY_OFFLINE_MODE:
            return None
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/inference-stats/metadata"),
            params=[("api_key", api_key), ("nocache", "true")],
        )
        payload = {
            "data": [
                {
                    "inference_ids": inference_ids,
                    "field_name": field_name,
                    "field_value": field_value,
                }
            ]
        }
        _translate_platform_api_errors(lambda: self._post_to_api(url, json=payload))

    @_records_api_call("register_image_at_roboflow")
    def register_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        image_bytes: bytes,
        batch_name: str,
        tags: Optional[List[str]] = None,
        inference_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        _refuse_when_offline(operation="register image")
        params = [("api_key", api_key), ("batch", batch_name)]
        if inference_id is not None:
            params.append(("inference_id", inference_id))
        for tag in tags if tags is not None else []:
            params.append(("tag", tag))
        url = _add_params_to_url(
            url=_api_url(f"dataset/{dataset_id}/upload"), params=params
        )
        data = {"name": f"{local_image_id}.jpg"}
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        files = {"file": ("imageToUpload", image_bytes, "image/jpeg")}
        parsed_response = _translate_platform_api_errors(
            lambda: self._post_to_api(url, data=data, files=files).json()
        )
        if not parsed_response.get("duplicate") and not parsed_response.get("success"):
            raise RoboflowAPIUnsuccessfulRequestError(
                f"Server rejected image: {parsed_response}"
            )
        return parsed_response

    @_records_api_call("annotate_image_at_roboflow")
    def annotate_image_at_roboflow(
        self,
        api_key: str,
        dataset_id: str,
        local_image_id: str,
        roboflow_image_id: str,
        annotation_content: str,
        annotation_file_type: str,
        is_prediction: bool = True,
    ) -> dict:
        _refuse_when_offline(operation="annotate image")
        url = _add_params_to_url(
            url=_api_url(f"dataset/{dataset_id}/annotate/{roboflow_image_id}"),
            params=[
                ("api_key", api_key),
                ("name", f"{local_image_id}.{annotation_file_type}"),
                ("prediction", str(is_prediction).lower()),
            ],
        )
        headers = self.build_api_headers(
            explicit_headers={"Content-Type": "text/plain"}
        )
        parsed_response = _translate_platform_api_errors(
            lambda: self._post_to_api(
                url, headers=headers, data=annotation_content
            ).json(),
            http_error_overrides={
                409: (
                    RoboflowAPIUnsuccessfulRequestError,
                    "Given datapoint already has annotation.",
                )
            },
        )
        if "error" in parsed_response or not parsed_response.get("success"):
            raise RoboflowAPIUnsuccessfulRequestError(
                f"Failed to save annotation for {roboflow_image_id}. "
                f"API response: {parsed_response}"
            )
        return parsed_response

    @_records_api_call("update_image_metadata_at_roboflow")
    def update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        image_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        add_tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        _refuse_when_offline(operation="update image metadata")
        payload: Dict[str, Any] = {}
        if metadata is not None:
            payload["metadata"] = metadata
        if add_tags is not None:
            payload["addTags"] = add_tags
        encoded_image_id = urllib.parse.quote(image_id, safe="")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/images/{encoded_image_id}/metadata"),
            params=[("api_key", api_key)],
        )
        return _translate_platform_api_errors(
            lambda: self._post_to_api(url, json=payload).json()
        )

    @_records_api_call("batch_update_image_metadata_at_roboflow")
    def batch_update_image_metadata_at_roboflow(
        self,
        api_key: str,
        workspace_id: str,
        updates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        _refuse_when_offline(operation="update image metadata")
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/images/metadata"),
            params=[("api_key", api_key)],
        )
        return _translate_platform_api_errors(
            lambda: self._post_to_api(url, json={"updates": updates}).json()
        )

    @_records_api_call("_make_request")
    def search_project_images_at_roboflow(
        self,
        api_key: str,
        workspace: str,
        project: str,
        image_base64: str,
        limit: int,
        fields: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        _refuse_when_offline(operation="search project images")
        payload = {
            "image_base64": image_base64,
            "limit": limit,
            "fields": fields or list(_SEARCH_DEFAULT_FIELDS),
        }
        url = _add_params_to_url(
            url=_api_url(f"{workspace}/{project}/search"),
            params=[("api_key", api_key)] if api_key and api_key != "local" else [],
        )
        return _translate_platform_api_errors(
            lambda: self._post_to_api(url, json=payload).json()
        )

    @_records_api_call("send_inference_results_to_model_monitoring")
    def send_inference_results_to_model_monitoring(
        self,
        api_key: str,
        workspace_id: str,
        inference_data: dict,
    ) -> None:
        if configuration.LEGACY_OFFLINE_MODE:
            return None
        url = _add_params_to_url(
            url=_api_url(f"{workspace_id}/inference-stats"),
            params=[("api_key", api_key)],
        )
        _translate_platform_api_errors(
            lambda: self._post_to_api(url, json=inference_data)
        )

    def get_device_id(self) -> Optional[str]:
        return configuration.DEVICE_ID

    def get_server_version(self) -> str:
        return configuration.SERVER_VERSION

    def get_system_info(self) -> dict:
        return collect_system_info()


class ServerWorkspaceResolver:
    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]:
        if not api_key or configuration.LEGACY_OFFLINE_MODE:
            return None

        try:
            workspace_id = PLATFORM_CLIENT.get_roboflow_workspace(api_key)
        except WorkspaceLoadError:
            return None
        except RoboflowAPIRequestError as error:
            raise error from None

        return workspace_id


class ServerImageCodec(WorkflowsLocalImageCodec):
    def __init__(self) -> None:
        self._loop_bridge: Optional[LoopBridge] = None

    def bind_loop(self, loop_bridge: LoopBridge) -> None:
        self._loop_bridge = loop_bridge

    def load_image(
        self, value: Any, disable_preproc_auto_orient: bool = False
    ) -> Tuple[np.ndarray, bool]:
        flags = choose_image_decoding_flags(
            disable_preproc_auto_orient=disable_preproc_auto_orient
        )
        if isinstance(value, dict) and "type" in value and "value" in value:
            if value["type"] == "url":
                return (
                    convert_gray_image_to_bgr(
                        self.fetch_url(value["value"], cv_imread_flags=flags)
                    ),
                    True,
                )
            if value["type"] == "file":
                return (
                    convert_gray_image_to_bgr(
                        self._read_local_file(value["value"], cv_imread_flags=flags)
                    ),
                    True,
                )
        elif isinstance(value, str) and value.startswith("http"):
            return (
                convert_gray_image_to_bgr(self.fetch_url(value, cv_imread_flags=flags)),
                True,
            )
        return super().load_image(
            value, disable_preproc_auto_orient=disable_preproc_auto_orient
        )

    def fetch_url(
        self, value: str, cv_imread_flags: int = cv2.IMREAD_COLOR
    ) -> np.ndarray:
        if configuration.LEGACY_OFFLINE_MODE:
            raise WorkflowImageLoadError(
                public_message="Loading images from a URL is not available while "
                "OFFLINE_MODE is enabled.",
                context=_IMAGE_LOADING_CONTEXT,
            )
        if not configuration.ALLOW_URL_INPUT:
            raise WorkflowImageLoadError(
                public_message="Loading images from a URL is disabled on this server.",
                context=_IMAGE_LOADING_CONTEXT,
            )
        if self._loop_bridge is None:
            raise RuntimeError("codec loop not bound")
        images, error = self._loop_bridge.run(
            legacy_bridge.fetch_url_images([value]),
            timeout=_URL_FETCH_BRIDGE_TIMEOUT_S,
        )
        if error is not None or not images:
            raise WorkflowImageLoadError(
                public_message="Could not fetch image from the given URL.",
                context=_IMAGE_LOADING_CONTEXT,
            )
        return decode_encoded_image_bytes(images[0], cv_imread_flags=cv_imread_flags)

    def decode_string(
        self,
        value: Union[str, bytes, bytearray],
        cv_imread_flags: int = cv2.IMREAD_COLOR,
    ) -> Tuple[np.ndarray, bool]:
        try:
            decoded = super().decode_string(value, cv_imread_flags=cv_imread_flags)
        except WorkflowImageLoadError as error:
            raise WorkflowImageLoadError(
                public_message=_NUMPY_INPUT_REFUSAL,
                context=_IMAGE_LOADING_CONTEXT,
            ) from error

        return decoded

    def ensure_local_file_load_allowed(self, path: str) -> None:
        if not configuration.ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM:
            raise WorkflowImageLoadError(
                public_message="Loading images from the local filesystem is disabled "
                "on this server.",
                context=_IMAGE_LOADING_CONTEXT,
            )

    def _read_local_file(self, path: str, cv_imread_flags: int) -> np.ndarray:
        self.ensure_local_file_load_allowed(path)
        image = cv2.imread(path, cv_imread_flags)
        if image is None:
            raise WorkflowImageLoadError(
                public_message=f"Could not load image from the local file: {path}",
                context=_IMAGE_LOADING_CONTEXT,
            )
        return image


PLATFORM_CLIENT = ServerRoboflowPlatformClient()
WORKSPACE_RESOLVER = ServerWorkspaceResolver()
GUARDED_IMAGE_CODEC = ServerImageCodec()
WORKFLOWS_CACHE = build_workflows_cache()


def bind_image_codec(init_parameters: Dict[str, Any]) -> None:
    set_image_codec(
        init_parameters.setdefault("workflows_core.image_codec", GUARDED_IMAGE_CODEC)
    )


def workflows_platform_bindings() -> Dict[str, Any]:
    return {
        "workflows_core.cache": WORKFLOWS_CACHE,
        "workflows_core.platform_client": PLATFORM_CLIENT,
        "workflows_core.workspace_resolver": WORKSPACE_RESOLVER,
        "workflows_core.inner_workflow_spec_resolver": inner_workflow_spec_resolver,
    }


def _local_workflow_response(workflow_id: str) -> dict:
    if not re.match(r"^[\w\-]+$", workflow_id):
        raise ValueError("Invalid workflow id")
    cache_root = Path(configuration.MODEL_CACHE_DIR)
    local_dir = cache_root / "workflow" / "local"
    path = local_dir / f"{sha256(workflow_id.encode()).hexdigest()}.json"
    not_found = FileNotFoundError(f"Local workflow file not found: {path}")
    for entry in (cache_root / "workflow", local_dir, path):
        if os.path.islink(entry):
            raise not_found
    path_status = os.lstat(path)
    if not stat.S_ISREG(path_status.st_mode):
        raise not_found
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        descriptor_status = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_status.st_mode) or (
            path_status.st_dev,
            path_status.st_ino,
        ) != (descriptor_status.st_dev, descriptor_status.st_ino):
            raise not_found
        source = os.fdopen(descriptor, "r", encoding="utf-8")
        descriptor = -1
        with source:
            return {"workflow": json.load(source)}
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _workflow_fetch_failure(status_code: int) -> LegacyHTTPError:
    message = _WORKFLOW_FETCH_FAILURE_MESSAGES.get(status_code)
    if message is None:
        return LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE)

    return LegacyHTTPError(status_code, message)


_PLATFORM_UNREACHABLE_CAUSES = (
    requests.exceptions.ConnectionError,
    ConnectionError,
    requests.exceptions.Timeout,
)


def _fetch_workflow_response(
    api_key: Optional[str],
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
) -> dict:
    if configuration.LEGACY_OFFLINE_MODE:
        raise LegacyHTTPError(
            503, "Internal error. Could not connect to Roboflow API."
        ) from ConnectionError("OFFLINE_MODE is enabled - cannot make API requests.")
    params: List[Tuple[str, str]] = []
    if api_key:
        params.append(("api_key", api_key))
    if workflow_version_id is not None:
        params.append(("workflow_version", workflow_version_id))
    url = _add_params_to_url(
        url=f"{configuration.API_BASE_URL.rstrip('/')}/{workspace_id}/workflows/{workflow_id}",
        params=params,
    )
    response = _platform_request(
        "get",
        PLATFORM_CLIENT.wrap_url(url),
        headers=PLATFORM_CLIENT.build_api_headers(),
        timeout=API_REQUEST_TIMEOUT_S,
    )
    if not _is_successful(response):
        raise _workflow_fetch_failure(response.status_code)
    try:
        payload = response.json()
    except ValueError as error:
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE) from error

    return payload


def _fetch_workflow_response_with_file_cache(
    api_key: Optional[str],
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
) -> dict:
    try:
        response = _fetch_workflow_response(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
    except LegacyHTTPError as error:
        if (
            not configuration.USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS
            or not isinstance(error.__cause__, _PLATFORM_UNREACHABLE_CAUSES)
        ):
            raise
        cached_response = definition_cache.load_definition(
            workspace_id,
            workflow_id,
            api_key=api_key,
            workflow_version_id=workflow_version_id,
        )
        if cached_response is None:
            raise
        return cached_response

    if configuration.USE_FILE_CACHE_FOR_WORKFLOWS_DEFINITIONS:
        definition_cache.store_definition(
            workspace_id,
            workflow_id,
            api_key=api_key,
            workflow_version_id=workflow_version_id,
            response=response,
        )

    return response


def _try_read_definition_cache(cache_key: str) -> Optional[dict]:
    try:
        return WORKFLOWS_CACHE.get(cache_key)
    except Exception as error:
        logger.warning(
            "Workflow definition cache unavailable, fetching from Roboflow API: %s",
            type(error).__name__,
        )
        return None


def _try_write_definition_cache(cache_key: str, specification: dict) -> None:
    try:
        WORKFLOWS_CACHE.set(
            cache_key,
            specification,
            expire=configuration.WORKFLOWS_DEFINITION_CACHE_TTL_S,
        )
    except Exception as error:
        logger.warning("Failed to cache workflow definition: %s", type(error).__name__)


@_records_api_call("get_workflow_specification")
def get_workflow_specification(
    api_key: Optional[str],
    workspace_id: str,
    workflow_id: str,
    use_cache: bool = True,
    workflow_version_id: Optional[str] = None,
) -> dict:
    cache_key = (
        f"workflow_definition:{workspace_id}:{workflow_id}:{workflow_version_id}:"
        f"{sha256((api_key or '').encode()).hexdigest()}"
    )
    if use_cache:
        cached = _try_read_definition_cache(cache_key)
        if cached:
            return cached
    if workspace_id == "local":
        response = _local_workflow_response(workflow_id)
    else:
        response = _fetch_workflow_response_with_file_cache(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
    if "workflow" not in response or "config" not in response["workflow"]:
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE)
    try:
        raw_config = response["workflow"]["config"]
        config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
        specification = config["specification"]
        if not isinstance(specification, dict):
            raise TypeError("Workflow specification must be a dictionary")
    except (KeyError, TypeError, ValueError) as error:
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE) from error
    specification["id"] = response["workflow"].get("id")
    if use_cache:
        _try_write_definition_cache(cache_key, specification)
    return specification


def inner_workflow_spec_resolver(
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
    init_parameters: Dict[str, Any],
) -> Dict[str, Any]:
    api_key = init_parameters.get("workflows_core.api_key")
    if workspace_id != "local" and not api_key:
        raise WorkflowDefinitionError(
            public_message=(
                "Resolving an `inner_workflow` step by workflow id requires a Roboflow API key. "
                "Set `workflows_core.api_key` in workflow init_parameters, inject "
                "`workflows_core.inner_workflow_spec_resolver`, or use "
                '`workflow_workspace_id` `"local"` with a matching on-disk workflow '
                "definition."
            ),
            context="workflow_compilation | inner_workflow_spec_resolution",
        )
    return get_workflow_specification(
        api_key=api_key,
        workspace_id=workspace_id,
        workflow_id=workflow_id,
        workflow_version_id=workflow_version_id,
    )


def _client_caused(
    step_name: str, status_code: int, message: str, error: Exception, context: str
) -> None:
    raise ClientCausedStepExecutionError(
        block_id=step_name,
        status_code=status_code,
        public_message=message,
        context=context,
        inner_error=error,
    ) from error


def _runtime_limited(step_name: str, message: str, error: Exception) -> None:
    raise RuntimeLimitsCausedStepExecutionError(
        block_id=step_name,
        status_code=507,
        public_message=message,
        context=_STEP_EXECUTION_CONTEXT,
        inner_error=error,
    ) from error


def step_error_handler(step_name: str, error: Exception) -> None:
    if isinstance(error, FeatureDeprecatedError):
        _client_caused(
            step_name,
            410,
            str(error),
            error,
            "workflow_execution | step_execution | feature_deprecated",
        )
    if isinstance(error, (ModelNotReadyError, ServerBusyError)):
        raise error
    if isinstance(error, LookupError) and isinstance(
        error.__cause__, InvalidPipelineIdError
    ):
        _client_caused(
            step_name,
            400,
            f"Problem with Workflow Block configuration - {error}",
            error.__cause__,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, ModelPackageRestrictedError) or (
        isinstance(error, ModelPackageAlternativesExhaustedError)
        and any(
            isinstance(alternative_error, ModelPackageRestrictedError)
            for alternative_error in error.alternatives_errors or []
        )
    ):
        _runtime_limited(
            step_name,
            "Model loading failed due to restrictions of server configuration - "
            "usually due to excessive runtime memory requirement of the model (for "
            "instance caused by large input size).",
            error,
        )
    if isinstance(error, (RoboflowAPINotAuthorizedError, UnauthorizedModelAccessError)):
        _client_caused(
            step_name,
            401,
            f"Unauthorized error occurred while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with "
            f"Roboflow API key.",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, PaymentRequiredError):
        _client_caused(
            step_name,
            402,
            f"Not enough credits to execute step {step_name}. "
            f"Verify your workspace billing page. Details: {error}",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, RoboflowAPIForbiddenError):
        _client_caused(
            step_name,
            403,
            f"Forbidden error occurred while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with "
            f"Roboflow API key.",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, RoboflowAPIUsagePausedError):
        _client_caused(
            step_name,
            423,
            f"Roboflow API usage is paused while executing step {step_name}. "
            f"Contact your workspace administrator to re-enable API keys. "
            f"Details: {error}",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, ModelRetrievalError):
        status_code = getattr(error, "status_code", None)
        if status_code in _MODEL_ACCESS_ERROR_MESSAGES:
            _client_caused(
                step_name,
                status_code,
                f"{_MODEL_ACCESS_ERROR_MESSAGES[status_code].format(step_name=step_name)} "
                f"Details: {error}",
                error,
                _STEP_EXECUTION_CONTEXT,
            )
    if isinstance(error, (RoboflowAPINotNotFoundError, ModelNotFoundError)):
        _client_caused(
            step_name,
            404,
            f"Could not find requested Roboflow resource while execution of step "
            f"{step_name} - details of error: {error}. This error usually mean the "
            f"problem with not existing model.",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, LegacyHTTPError):
        if error.status_code == 507:
            _runtime_limited(step_name, error.message, error)
        if (
            error.status_code < 500
            or error.status_code == 501
            or isinstance(error, ImageFetchError)
        ):
            _client_caused(
                step_name,
                error.status_code,
                error.message,
                error,
                _STEP_EXECUTION_CONTEXT,
            )
        return None
    if isinstance(error, HTTPCallErrorError):
        return _handle_remote_call_error(step_name, error)
    return None


_REMOTE_CALL_ERROR_MESSAGES = {
    400: "Bad request error detected while remote execution of step {step_name} - "
    "details of error: {error}. This error usually mean that the Workflow block "
    "configuration is faulty.",
    401: "Unauthorized error occurred while remote execution of step {step_name} - "
    "details of error: {error}. This error usually mean the problem with Roboflow "
    "API key.",
    402: "Not enough credits to remote execute step {step_name}. Verify your "
    "workspace billing page. Details: {error}",
    403: "Forbidden error occurred while remote execution of step {step_name} - "
    "details of error: {error}. This error usually mean the problem with Roboflow "
    "API key.",
    404: "Could not find requested Roboflow resource while remote execution of step "
    "{step_name} - details of error: {error}. This error usually mean the problem "
    "with not existing model.",
    410: "Deprecated feature usage detected while remote execution of step "
    "{step_name} - details of error: {error}.",
    423: "Roboflow API usage is paused while remote executing step {step_name}. "
    "Contact your workspace administrator to re-enable API keys. Details: {error}",
}


def _handle_remote_call_error(step_name: str, error: HTTPCallErrorError) -> None:
    if error.status_code == 507:
        _runtime_limited(
            step_name,
            f"Could not complete workflow execution due to configured runtime "
            f"constraints. Details: {error.api_message}",
            error,
        )
    if error.status_code == 501:
        public_message = (
            error.api_message
            or f"Remote execution of step {step_name} is not supported on this deployment."
        )
        raise ClientCausedStepExecutionError(
            block_id=step_name,
            status_code=501,
            public_message=public_message,
            context="workflow_execution | step_execution | deployment_not_supported",
            inner_error=ModelDeploymentNotSupportedError(public_message),
        ) from error
    message = _REMOTE_CALL_ERROR_MESSAGES.get(error.status_code)
    if message is None:
        return None
    _client_caused(
        step_name,
        error.status_code,
        message.format(step_name=step_name, error=error),
        error,
        (
            "workflow_execution | step_execution | feature_deprecated"
            if error.status_code == 410
            else _STEP_EXECUTION_CONTEXT
        ),
    )
