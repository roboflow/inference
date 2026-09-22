"""Importing this module installs the process-wide `WorkflowsConfiguration`, which must
happen before anything imports `roboflow_workflows.environment`."""

from __future__ import annotations

import json
import logging
import os
import re
import warnings
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
    offline_mode = get_boolean_from_env("OFFLINE_MODE", default=False)
    secure_gateway = os.environ.get("SECURE_GATEWAY") or None
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
            gcp_serverless=False,
            lambda_runtime=False,
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


SERVER_WORKFLOWS_CONFIGURATION = build_workflows_configuration()
configure_process(SERVER_WORKFLOWS_CONFIGURATION)

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
)
from roboflow_workflows.utils.image_encoding import (  # noqa: E402
    choose_image_decoding_flags,
    convert_gray_image_to_bgr,
    decode_encoded_image_bytes,
)
from roboflow_workflows.utils.in_memory_cache import (  # noqa: E402
    InMemoryWorkflowsCache,
)

from inference_models.errors import (  # noqa: E402
    ModelNotFoundError,
    ModelPackageRestrictedError,
    ModelRetrievalError,
    UnauthorizedModelAccessError,
)
from inference_models.weights_providers.roboflow import (  # noqa: E402
    roboflow_secure_gateway_proxy_url_builder,
)
from inference_sdk.http.errors import HTTPCallErrorError  # noqa: E402
from inference_server.framework.input_parsers.url_fetch import (  # noqa: E402
    URL_FETCH_TIMEOUT_S,
)
from inference_server.legacy import bridge as legacy_bridge  # noqa: E402
from inference_server.legacy.bridge import LoopBridge  # noqa: E402
from inference_server.legacy.errors import LegacyHTTPError  # noqa: E402

API_REQUEST_TIMEOUT_S = get_float_from_env(
    "ROBOFLOW_API_REQUEST_TIMEOUT", default=120.0
)
_URL_FETCH_BRIDGE_TIMEOUT_S = URL_FETCH_TIMEOUT_S + 5
_IMAGE_LOADING_CONTEXT = "workflow_execution | image_loading"
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
    import urllib.parse

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


class ServerRoboflowPlatformClient:
    def post(
        self,
        endpoint: str,
        api_key: Optional[str],
        payload: Optional[dict] = None,
        params: Optional[List[Tuple[str, str]]] = None,
        http_errors_handlers: Optional[HttpErrorHandlers] = None,
    ) -> dict:
        url_params: List[Tuple[str, str]] = []
        if api_key:
            url_params.append(("api_key", api_key))
        if params:
            url_params.extend(params)
        url = _add_params_to_url(
            url=f"{configuration.API_BASE_URL.rstrip('/')}/{endpoint.strip('/')}",
            params=url_params,
        )
        response = requests.post(
            url=self.wrap_url(url),
            json=payload,
            headers=self.build_api_headers(),
            timeout=API_REQUEST_TIMEOUT_S,
        )
        if not _is_successful(response):
            message = _api_error_message(response, api_key)
            handler = (http_errors_handlers or {}).get(response.status_code)
            if handler is not None:
                handler(requests.exceptions.HTTPError(message, response=response))
            raise LegacyHTTPError(response.status_code, message)
        return response.json()

    def build_api_headers(
        self, explicit_headers: Optional[Dict[str, Union[str, List[str]]]] = None
    ) -> Dict[str, Union[str, List[str]]]:
        headers: Dict[str, Union[str, List[str]]] = {
            "x-roboflow-inference-version": configuration.SERVER_VERSION,
            "x-allow-chunked-response": "true",
        }
        if configuration.ROBOFLOW_API_EXTRA_HEADERS:
            try:
                headers.update(json.loads(configuration.ROBOFLOW_API_EXTRA_HEADERS))
            except ValueError:
                logger.warning("Could not decode ROBOFLOW_API_EXTRA_HEADERS")
        headers.update(explicit_headers or {})
        return headers

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
        return roboflow_secure_gateway_proxy_url_builder(url, None)


class ServerWorkspaceResolver:
    def resolve_workspace(self, api_key: Optional[str]) -> Optional[str]:
        if not api_key:
            return None
        url = _add_params_to_url(
            url=f"{configuration.API_BASE_URL.rstrip('/')}/",
            params=[("api_key", api_key), ("nocache", "true")],
        )
        try:
            response = requests.get(
                url=PLATFORM_CLIENT.wrap_url(url),
                headers=PLATFORM_CLIENT.build_api_headers(),
                timeout=API_REQUEST_TIMEOUT_S,
            )
            if not _is_successful(response):
                return None
            workspace_id = response.json().get("workspace")
        except Exception:
            logger.warning("Could not resolve Roboflow workspace", exc_info=True)
            return None
        if not isinstance(workspace_id, str) or not workspace_id:
            return None
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
        if configuration.OFFLINE_MODE:
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
            legacy_bridge.fetch_images_from_urls([value]),
            timeout=_URL_FETCH_BRIDGE_TIMEOUT_S,
        )
        if error is not None or not images:
            raise WorkflowImageLoadError(
                public_message="Could not fetch image from the given URL.",
                context=_IMAGE_LOADING_CONTEXT,
            )
        return decode_encoded_image_bytes(images[0], cv_imread_flags=cv_imread_flags)

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
WORKFLOWS_CACHE = InMemoryWorkflowsCache()


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
    path = (
        Path(configuration.MODEL_CACHE_DIR)
        / "workflow"
        / "local"
        / f"{sha256(workflow_id.encode()).hexdigest()}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Local workflow file not found: {path}")
    with open(path, "r") as source:
        return {"workflow": json.load(source)}


def _fetch_workflow_response(
    api_key: Optional[str],
    workspace_id: str,
    workflow_id: str,
    workflow_version_id: Optional[str],
) -> dict:
    params: List[Tuple[str, str]] = []
    if api_key:
        params.append(("api_key", api_key))
    if workflow_version_id is not None:
        params.append(("workflow_version", workflow_version_id))
    url = _add_params_to_url(
        url=f"{configuration.API_BASE_URL.rstrip('/')}/{workspace_id}/workflows/{workflow_id}",
        params=params,
    )
    response = requests.get(
        url=PLATFORM_CLIENT.wrap_url(url),
        headers=PLATFORM_CLIENT.build_api_headers(),
        timeout=API_REQUEST_TIMEOUT_S,
    )
    if not _is_successful(response):
        raise LegacyHTTPError(
            response.status_code, _api_error_message(response, api_key)
        )
    return response.json()


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
        cached = WORKFLOWS_CACHE.get(cache_key)
        if cached:
            return cached
    if workspace_id == "local":
        response = _local_workflow_response(workflow_id)
    else:
        response = _fetch_workflow_response(
            api_key=api_key,
            workspace_id=workspace_id,
            workflow_id=workflow_id,
            workflow_version_id=workflow_version_id,
        )
    try:
        raw_config = response["workflow"]["config"]
        config = json.loads(raw_config) if isinstance(raw_config, str) else raw_config
        specification = config["specification"]
        if not isinstance(specification, dict):
            raise TypeError("Workflow specification must be a dictionary")
    except (KeyError, TypeError, ValueError) as error:
        raise LegacyHTTPError(
            502, "Could not find workflow specification in API response"
        ) from error
    specification["id"] = response["workflow"].get("id")
    if use_cache:
        WORKFLOWS_CACHE.set(
            cache_key,
            specification,
            expire=configuration.WORKFLOWS_DEFINITION_CACHE_TTL_S,
        )
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
    if isinstance(error, UnauthorizedModelAccessError):
        _client_caused(
            step_name,
            401,
            f"Unauthorized error occurred while execution of step {step_name} - "
            f"details of error: {error}. This error usually mean the problem with "
            f"Roboflow API key.",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, ModelNotFoundError):
        _client_caused(
            step_name,
            404,
            f"Could not find requested Roboflow resource while execution of step "
            f"{step_name} - details of error: {error}. This error usually mean the "
            f"problem with not existing model.",
            error,
            _STEP_EXECUTION_CONTEXT,
        )
    if isinstance(error, ModelPackageRestrictedError):
        _runtime_limited(
            step_name,
            "Model loading failed due to restrictions of server configuration - "
            "usually due to excessive runtime memory requirement of the model (for "
            "instance caused by large input size).",
            error,
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
    if isinstance(error, LegacyHTTPError):
        if error.status_code == 507:
            _runtime_limited(step_name, error.message, error)
        _client_caused(
            step_name,
            error.status_code,
            error.message,
            error,
            _STEP_EXECUTION_CONTEXT,
        )
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
        _client_caused(
            step_name,
            501,
            error.api_message
            or f"Remote execution of step {step_name} is not supported on this deployment.",
            error,
            "workflow_execution | step_execution | deployment_not_supported",
        )
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
