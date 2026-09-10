"""The installed `WorkflowsConfiguration`, bound as module constants.

This is the workflows-side replacement for `inference.core.env`: the same
UPPER_CASE names, the same types, the same values - sourced from the
configuration object the host installed with
`inference.core.workflows.configuration.configure_process`, never from
`os.environ`.

The values are bound ONCE, at this module's import - exactly as
`inference/core/env.py` binds its own at its import. Importing this module
therefore freezes the process configuration; a later `configure_process` with
different values raises. Tests that need different values call
`configuration.reset_configuration()`, `configure_process(...)`, and
`importlib.reload` this module plus its consumer.
"""

from inference.core.workflows.configuration import get_configuration

_CONFIGURATION = get_configuration()

# --- engine ---
WORKFLOWS_STEP_EXECUTION_MODE = _CONFIGURATION.engine.step_execution_mode
WORKFLOWS_ASYNC_FUTURE_RESULT_TIMEOUT = (
    _CONFIGURATION.engine.async_future_result_timeout
)
WORKFLOWS_MAX_INNER_WORKFLOW_DEPTH = _CONFIGURATION.engine.max_inner_workflow_depth
WORKFLOWS_MAX_INNER_WORKFLOW_COUNT = _CONFIGURATION.engine.max_inner_workflow_count
ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS = (
    _CONFIGURATION.engine.allow_custom_python_execution
)
WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE = (
    _CONFIGURATION.engine.custom_python_execution_mode
)
ALLOW_WORKFLOW_BLOCKS_ACCESSING_LOCAL_STORAGE = (
    _CONFIGURATION.engine.allow_blocks_accessing_local_storage
)
ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES = (
    _CONFIGURATION.engine.allow_blocks_accessing_environmental_variables
)
WORKFLOW_BLOCKS_WRITE_DIRECTORY = _CONFIGURATION.engine.blocks_write_directory
WORKFLOW_DISABLED_BLOCK_TYPES = list(_CONFIGURATION.engine.disabled_block_types)
WORKFLOW_DISABLED_BLOCK_PATTERNS = list(_CONFIGURATION.engine.disabled_block_patterns)

# --- tensor representation ---
ENABLE_TENSOR_DATA_REPRESENTATION = _CONFIGURATION.tensor.representation_enabled
WORKFLOWS_IMAGE_TENSOR_DEVICE = _CONFIGURATION.tensor.image_tensor_device
WORKFLOWS_TENSOR_VISUALISATION_VALIDATE_OWNERS = (
    _CONFIGURATION.tensor.visualisation_validate_owners
)
WORKFLOWS_SAM_VIDEO_MASK_REPRESENTATION = (
    _CONFIGURATION.tensor.sam_video_mask_representation
)
WORKFLOWS_ENFORCE_DENSE_INSTANCE_MASKS = (
    _CONFIGURATION.tensor.enforce_dense_instance_masks
)

# --- remote step execution ---
WORKFLOWS_REMOTE_API_TARGET = _CONFIGURATION.remote.api_target
WORKFLOWS_REMOTE_API_KEY_TRANSPORT = _CONFIGURATION.remote.api_key_transport
LOCAL_INFERENCE_API_URL = _CONFIGURATION.remote.local_inference_api_url
HOSTED_DETECT_URL = _CONFIGURATION.remote.hosted_detect_url
HOSTED_CLASSIFICATION_URL = _CONFIGURATION.remote.hosted_classification_url
HOSTED_INSTANCE_SEGMENTATION_URL = (
    _CONFIGURATION.remote.hosted_instance_segmentation_url
)
HOSTED_SEMANTIC_SEGMENTATION_URL = (
    _CONFIGURATION.remote.hosted_semantic_segmentation_url
)
HOSTED_CORE_MODEL_URL = _CONFIGURATION.remote.hosted_core_model_url
WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE = (
    _CONFIGURATION.remote.max_step_batch_size
)
WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS = (
    _CONFIGURATION.remote.max_step_concurrent_requests
)

# --- platform / deployment ---
API_BASE_URL = _CONFIGURATION.platform.api_base_url
OFFLINE_MODE = _CONFIGURATION.platform.offline_mode
SECURE_GATEWAY = _CONFIGURATION.platform.secure_gateway
GCP_SERVERLESS = _CONFIGURATION.platform.gcp_serverless

# --- fonts ---
ALLOW_WORKFLOWS_FONTS_DOWNLOAD = _CONFIGURATION.fonts.allow_download
MODEL_CACHE_DIR = _CONFIGURATION.fonts.model_cache_dir

# --- model feature flags ---
LMM_ENABLED = _CONFIGURATION.models.lmm_enabled
CLIP_VERSION_ID = _CONFIGURATION.models.clip_version_id
CORE_MODEL_SAM2_ENABLED = _CONFIGURATION.models.core_model_sam2_enabled
CORE_MODEL_SAM3_ENABLED = _CONFIGURATION.models.core_model_sam3_enabled
CORE_MODEL_PE_ENABLED = _CONFIGURATION.models.core_model_pe_enabled
CORE_MODEL_GAZE_ENABLED = _CONFIGURATION.models.core_model_gaze_enabled
SAM3_EXEC_MODE = _CONFIGURATION.models.sam3_exec_mode
SAM3_3D_OBJECTS_ENABLED = _CONFIGURATION.models.sam3_3d_objects_enabled
FLORENCE2_ENABLED = _CONFIGURATION.models.florence2_enabled
QWEN_2_5_ENABLED = _CONFIGURATION.models.qwen_2_5_enabled
QWEN_3_ENABLED = _CONFIGURATION.models.qwen_3_enabled
QWEN_3_5_ENABLED = _CONFIGURATION.models.qwen_3_5_enabled
SMOLVLM2_ENABLED = _CONFIGURATION.models.smolvlm2_enabled
MOONDREAM2_ENABLED = _CONFIGURATION.models.moondream2_enabled
DEPTH_ESTIMATION_ENABLED = _CONFIGURATION.models.depth_estimation_enabled
COSMOS3_ENABLED = _CONFIGURATION.models.cosmos3_enabled
GLM_OCR_ENABLED = _CONFIGURATION.models.glm_ocr_enabled

# --- modal / webexec custom-python backend ---
MODAL_TOKEN_ID = _CONFIGURATION.modal.token_id
MODAL_TOKEN_SECRET = _CONFIGURATION.modal.token_secret
MODAL_WORKSPACE_NAME = _CONFIGURATION.modal.workspace_name
MODAL_ALLOW_ANONYMOUS_EXECUTION = _CONFIGURATION.modal.allow_anonymous_execution
MODAL_ANONYMOUS_WORKSPACE_NAME = _CONFIGURATION.modal.anonymous_workspace_name
WEBEXEC_MODAL_APP_NAME = _CONFIGURATION.modal.app_name
WEBEXEC_MODAL_EXECUTOR_IDLE_TTL_SECONDS = _CONFIGURATION.modal.executor_idle_ttl_seconds
WEBEXEC_JPEG_QUALITY = _CONFIGURATION.modal.jpeg_quality
WEBEXEC_TRANSPORT = _CONFIGURATION.modal.transport
WEBEXEC_WS_CONNECT_TIMEOUT_SECONDS = _CONFIGURATION.modal.ws_connect_timeout_seconds
WEBEXEC_WS_READ_TIMEOUT_SECONDS = _CONFIGURATION.modal.ws_read_timeout_seconds
WEBEXEC_WS_CONNECTION_POOL_SIZE = _CONFIGURATION.modal.ws_connection_pool_size
WEBEXEC_WS_FAIL_ON_SESSION_LOSS = _CONFIGURATION.modal.ws_fail_on_session_loss
WEBEXEC_WS_IDLE_RELEASE_SECONDS = _CONFIGURATION.modal.ws_idle_release_seconds

# --- secrets ---
API_KEY = _CONFIGURATION.secrets.api_key
ROBOFLOW_INTERNAL_SERVICE_NAME = _CONFIGURATION.secrets.roboflow_internal_service_name
ROBOFLOW_INTERNAL_SERVICE_SECRET = (
    _CONFIGURATION.secrets.roboflow_internal_service_secret
)

# --- debug ---
INFERENCE_DEBUG_OUTPUT_DIR = _CONFIGURATION.debug.output_dir
