"""Builds the Workflows configuration from the server's resolved settings.

`inference/core/env.py` does the real work - `str2bool`, `.lower()`, the
`OFFLINE_MODE` / `SECURE_GATEWAY` rewrites of `WORKFLOWS_STEP_EXECUTION_MODE`
(`env.py:905-931`), `WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE` (`:953-960`) and
`SAM3_EXEC_MODE` (`:407-414`), the `PROJECT`-dependent hosted URLs, the
api-key-transport validation and the torch-device materialisation
(`:1524-1547`). This module ONLY copies the resolved attributes, so none of
that logic is duplicated and none of it can drift.

`install_workflows_configuration()` is called from `inference/core/__init__.py`.
The invariant is narrower than "before any workflows module": installation
precedes every import of the constants facade
(`inference.core.workflows.environment`) and of every configuration-consuming
workflows module, including `core_steps/loader.py`'s tensor-mode branches.
A handful of configuration-independent workflows modules are imported earlier,
on the bootstrap path (`inference.core.env` -> `utils/environment.py` ->
`core/exceptions.py` -> `workflows/prototypes/platform_errors.py`, plus
`workflows/configuration.py` itself, imported by this module); they never read
the facade and must stay that way, or they would freeze standalone defaults
into it before the server's configuration is installed.
"""

from typing import Optional

from inference.core import env
from inference.core.workflows.configuration import (
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
)

_SERVER_CONFIGURATION: Optional[WorkflowsConfiguration] = None


def build_configuration_from_env() -> WorkflowsConfiguration:
    return WorkflowsConfiguration(
        engine=EngineConfiguration(
            step_execution_mode=env.WORKFLOWS_STEP_EXECUTION_MODE,
            async_future_result_timeout=env.WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT,
            max_inner_workflow_depth=env.WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH,
            max_inner_workflow_count=env.WORKFLOWS_MAX_INNER_WORKFLOW_COUNT,
            allow_custom_python_execution=env.ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,
            custom_python_execution_mode=env.WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
            allow_blocks_accessing_local_storage=env.ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE,
            allow_blocks_accessing_environmental_variables=env.ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES,
            blocks_write_directory=env.WORKFLOW_BLOCKS_WRITE_DIRECTORY,
            disabled_block_types=tuple(env.WORKFLOW_DISABLED_BLOCK_TYPES),
            disabled_block_patterns=tuple(env.WORKFLOW_DISABLED_BLOCK_PATTERNS),
        ),
        tensor=TensorConfiguration(
            representation_enabled=env.ENABLE_TENSOR_DATA_REPRESENTATION,
            image_tensor_device=env.WORKFLOWS_IMAGE_TENSOR_DEVICE,
            visualisation_validate_owners=env.WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS,
            sam_video_mask_representation=env.WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION,
            enforce_dense_instance_masks=env.WORKFLOWS_ENFORCE_DENSE_INSTANCE_MASKS,
        ),
        remote=RemoteExecutionConfiguration(
            api_target=env.WORKFLOWS_REMOTE_API_TARGET,
            api_key_transport=env.WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
            local_inference_api_url=env.LOCAL_INFERENCE_API_URL,
            hosted_detect_url=env.HOSTED_DETECT_URL,
            hosted_classification_url=env.HOSTED_CLASSIFICATION_URL,
            hosted_instance_segmentation_url=env.HOSTED_INSTANCE_SEGMENTATION_URL,
            hosted_semantic_segmentation_url=env.HOSTED_SEMANTIC_SEGMENTATION_URL,
            hosted_core_model_url=env.HOSTED_CORE_MODEL_URL,
            max_step_batch_size=env.WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
            max_step_concurrent_requests=env.WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        ),
        platform=PlatformConfiguration(
            api_base_url=env.API_BASE_URL,
            offline_mode=env.OFFLINE_MODE,
            secure_gateway=env.SECURE_GATEWAY,
            gcp_serverless=env.GCP_SERVERLESS,
        ),
        fonts=FontsConfiguration(
            allow_download=env.ALLOW_WORKFLOWS_FONTS_DOWNLOAD,
            model_cache_dir=env.MODEL_CACHE_DIR,
        ),
        models=ModelsConfiguration(
            lmm_enabled=env.LMM_ENABLED,
            clip_version_id=env.CLIP_VERSION_ID,
            core_model_sam2_enabled=env.CORE_MODEL_SAM2_ENABLED,
            core_model_sam3_enabled=env.CORE_MODEL_SAM3_ENABLED,
            core_model_pe_enabled=env.CORE_MODEL_PE_ENABLED,
            core_model_gaze_enabled=env.CORE_MODEL_GAZE_ENABLED,
            sam3_exec_mode=env.SAM3_EXEC_MODE,
            sam3_3d_objects_enabled=env.SAM3_3D_OBJECTS_ENABLED,
            florence2_enabled=env.FLORENCE2_ENABLED,
            qwen_2_5_enabled=env.QWEN_2_5_ENABLED,
            qwen_3_enabled=env.QWEN_3_ENABLED,
            qwen_3_5_enabled=env.QWEN_3_5_ENABLED,
            smolvlm2_enabled=env.SMOLVLM2_ENABLED,
            moondream2_enabled=env.MOONDREAM2_ENABLED,
            depth_estimation_enabled=env.DEPTH_ESTIMATION_ENABLED,
            cosmos3_enabled=env.COSMOS3_ENABLED,
            glm_ocr_enabled=env.GLM_OCR_ENABLED,
        ),
        modal=ModalConfiguration(
            token_id=env.MODAL_TOKEN_ID,
            token_secret=env.MODAL_TOKEN_SECRET,
            workspace_name=env.MODAL_WORKSPACE_NAME,
            allow_anonymous_execution=env.MODAL_ALLOW_ANONYMOUS_EXECUTION,
            anonymous_workspace_name=env.MODAL_ANONYMOUS_WORKSPACE_NAME,
            app_name=env.WEBEXEC_MODAL_APP_NAME,
            executor_idle_ttl_seconds=env.WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS,
            jpeg_quality=env.WEBEXEC_JPEG_QUALITY,
            transport=env.WEBEXEC_TRANSPORT,
            ws_connect_timeout_seconds=env.WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS,
            ws_read_timeout_seconds=env.WEBEXEC_WS_READ_TIMEOUT_SECONDS,
            ws_connection_pool_size=env.WEBEXEC_WS_CONNECTION_POOL_SIZE,
            ws_fail_on_session_loss=env.WEBEXEC_WS_FAIL_ON_SESSION_LOSS,
            ws_idle_release_seconds=env.WEBEXEC_WS_IDLE_RELEASE_SECONDS,
        ),
        secrets=SecretsConfiguration(
            api_key=env.API_KEY,
            roboflow_internal_service_name=env.ROBOFLOW_INTERNAL_SERVICE_NAME,
            roboflow_internal_service_secret=env.ROBOFLOW_INTERNAL_SERVICE_SECRET,
        ),
        debug=DebugConfiguration(
            output_dir=env.INFERENCE_DEBUG_OUTPUT_DIR,
        ),
    )


def server_workflows_configuration() -> WorkflowsConfiguration:
    """The one configuration this process uses. Built once, reused thereafter.

    Memoised so every composition root passes the SAME object: the engine's
    `ensure_process_configuration_matches` then short-circuits on identity, and
    a repeated `install_workflows_configuration()` is a no-op.
    """
    global _SERVER_CONFIGURATION
    if _SERVER_CONFIGURATION is None:
        _SERVER_CONFIGURATION = build_configuration_from_env()
    return _SERVER_CONFIGURATION


def install_workflows_configuration() -> None:
    """Hand the configuration to workflows. Idempotent by object identity."""
    configure_process(server_workflows_configuration())
