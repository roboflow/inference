import asyncio
from pathlib import Path
from typing import Dict, Optional, Tuple

from inference.core import logger
from inference.core.env import (
    ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,
    INTERNAL_WEIGHTS_URL_SUFFIX,
    LOG_LEVEL,
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WORKSPACE_NAME,
    MODEL_CACHE_DIR,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
    PROJECT,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    WEBRTC_MODAL_APP_NAME,
    WEBRTC_MODAL_CLOUD,
    WEBRTC_MODAL_DEFAULT_REGION,
    WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
    WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
    WEBRTC_MODAL_FUNCTION_GPU,
    WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
    WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
    WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
    WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
    WEBRTC_MODAL_IMAGE_NAME,
    WEBRTC_MODAL_IMAGE_TAG,
    WEBRTC_MODAL_RESPONSE_TIMEOUT,
    WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
    WEBRTC_MODAL_SUPPORTED_REGIONS,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
)
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.webrtc import (
    init_rtc_peer_connection_with_loop,
)
from inference.core.version import __version__

try:
    import modal
except ImportError:
    modal = None


def _get_common_env_dict(gpu: Optional[str] = None) -> Dict[str, str]:
    """Get common environment variables for Modal functions."""
    return {
        "ROBOFLOW_INTERNAL_SERVICE_SECRET": ROBOFLOW_INTERNAL_SERVICE_SECRET,
        "ROBOFLOW_INTERNAL_SERVICE_NAME": WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
        "PROJECT": PROJECT,
        "LOG_LEVEL": LOG_LEVEL,
        "INTERNAL_WEIGHTS_URL_SUFFIX": INTERNAL_WEIGHTS_URL_SUFFIX,
        "MODELS_CACHE_AUTH_ENABLED": str(MODELS_CACHE_AUTH_ENABLED),
        "MODELS_CACHE_AUTH_CACHE_TTL": str(MODELS_CACHE_AUTH_CACHE_TTL),
        "MODELS_CACHE_AUTH_CACHE_MAX_SIZE": str(MODELS_CACHE_AUTH_CACHE_MAX_SIZE),
        "METRICS_ENABLED": "False",
        "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": str(
            ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
        ),
        "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
        "MODAL_TOKEN_ID": MODAL_TOKEN_ID,
        "MODAL_TOKEN_SECRET": MODAL_TOKEN_SECRET,
        "MODAL_WORKSPACE_NAME": MODAL_WORKSPACE_NAME,
        "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": "False",
        "DISABLE_VERSION_CHECK": "True",
        "MODEL_CACHE_DIR": MODEL_CACHE_DIR,
        "HF_HOME": Path(MODEL_CACHE_DIR).joinpath("hf_home").as_posix(),
        "TELEMETRY_USE_PERSISTENT_QUEUE": "False",
        "DISABLE_INFERENCE_CACHE": "True",
        "WEBRTC_MODAL_FUNCTION_GPU": gpu or "",
        "WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW": str(
            WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW
        ),
        "WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS": str(
            WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS
        ),
        "WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS": str(
            WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS
        ),
        "WEBRTC_MODAL_FUNCTION_TIME_LIMIT": str(WEBRTC_MODAL_FUNCTION_TIME_LIMIT),
        "WEBRTC_MODAL_IMAGE_NAME": WEBRTC_MODAL_IMAGE_NAME,
        "WEBRTC_MODAL_IMAGE_TAG": WEBRTC_MODAL_IMAGE_TAG,
        "WEBRTC_MODAL_RTSP_PLACEHOLDER": WEBRTC_MODAL_RTSP_PLACEHOLDER or "",
        "WEBRTC_MODAL_RTSP_PLACEHOLDER_URL": WEBRTC_MODAL_RTSP_PLACEHOLDER_URL or "",
        "ONNXRUNTIME_EXECUTION_PROVIDERS": (
            "CUDAExecutionProvider" if gpu else "CPUExecutionProvider"
        ),
    }


if modal is not None:
    # https://modal.com/docs/reference/modal.Image
    video_processing_image = (
        modal.Image.from_registry(
            f"{WEBRTC_MODAL_IMAGE_NAME}:{WEBRTC_MODAL_IMAGE_TAG if WEBRTC_MODAL_IMAGE_TAG else __version__}"
        )
        .pip_install("modal")
        .entrypoint([])
    )

    # https://modal.com/docs/reference/modal.Volume
    rfcache_volume = modal.Volume.from_name("rfcache", create_if_missing=True)

    # Single app for all WebRTC functions
    # https://modal.com/docs/reference/modal.App
    app = modal.App(
        name=WEBRTC_MODAL_APP_NAME,
        image=video_processing_image,
    )

    # Cache of created functions: key is (region, instance_type), value is the modal function
    _MODAL_FUNCTIONS_CACHE: Dict[Tuple[str, str], "modal.Function"] = {}

    def _create_function_for_config(region: str, instance_type: str) -> "modal.Function":
        """
        Create a Modal function for a specific region and instance type.
        
        Args:
            region: AWS/GCP region (e.g., 'us-east-1')
            instance_type: 'cpu' or 'gpu'
            
        Returns:
            Modal function decorated with appropriate configuration
        """
        # Determine GPU configuration
        gpu = None
        if instance_type == "gpu":
            gpu = WEBRTC_MODAL_FUNCTION_GPU or "any"

        # Create a unique function name
        func_name = f"rtc_peer_connection_{region.replace('-', '_')}_{instance_type}"

        # https://modal.com/docs/reference/modal.App#function
        @app.function(
            name=func_name,
            min_containers=WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
            buffer_containers=WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
            scaledown_window=WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
            timeout=WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
            enable_memory_snapshot=WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
            experimental_options=(
                {"enable_gpu_snapshot": True}
                if WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT and gpu
                else {}
            ),
            gpu=gpu,
            max_inputs=WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
            cloud=WEBRTC_MODAL_CLOUD,
            region=region,  # Region specified at function level
            env=_get_common_env_dict(gpu=gpu),
            volumes={MODEL_CACHE_DIR: rfcache_volume},
        )
        def rtc_peer_connection_modal(
            webrtc_request: WebRTCWorkerRequest,
            q: modal.Queue,
        ):
            logger.info(
                "Processing webrtc offer in region=%s, instance_type=%s",
                region,
                instance_type,
            )

            def send_answer(obj: WebRTCWorkerResult):
                logger.info("Sending webrtc answer")
                q.put(obj)

            asyncio.run(
                init_rtc_peer_connection_with_loop(
                    webrtc_request=webrtc_request,
                    send_answer=send_answer,
                )
            )

        return rtc_peer_connection_modal

    def _get_or_create_function(region: str, instance_type: str) -> "modal.Function":
        """Get or create a Modal function for the specified region and instance type."""
        cache_key = (region, instance_type)
        if cache_key not in _MODAL_FUNCTIONS_CACHE:
            logger.info(
                "Creating Modal function for region=%s, instance_type=%s",
                region,
                instance_type,
            )
            _MODAL_FUNCTIONS_CACHE[cache_key] = _create_function_for_config(
                region=region, instance_type=instance_type
            )
        return _MODAL_FUNCTIONS_CACHE[cache_key]

    def spawn_rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
    ) -> WebRTCWorkerResult:
        """
        Spawn a Modal function to handle WebRTC connection.

        The function will be deployed to the specified region and instance type
        from the webrtc_request, or use defaults if not specified.
        
        All functions are part of the same Modal app, but each function
        is configured for a specific region and instance type.
        """
        # Determine region and instance type from request or use defaults
        region = webrtc_request.region or WEBRTC_MODAL_DEFAULT_REGION
        instance_type = webrtc_request.instance_type or (
            "gpu" if WEBRTC_MODAL_FUNCTION_GPU else "cpu"
        )

        # Validate region is supported
        if region not in WEBRTC_MODAL_SUPPORTED_REGIONS:
            logger.warning(
                "Requested region %s not in supported regions %s, using default %s",
                region,
                WEBRTC_MODAL_SUPPORTED_REGIONS,
                WEBRTC_MODAL_DEFAULT_REGION,
            )
            region = WEBRTC_MODAL_DEFAULT_REGION

        # Validate instance type
        if instance_type not in ["cpu", "gpu"]:
            logger.warning(
                "Invalid instance type %s, defaulting to cpu", instance_type
            )
            instance_type = "cpu"

        logger.info(
            "Spawning WebRTC Modal function for region=%s, instance_type=%s",
            region,
            instance_type,
        )

        # Get or create the function for this configuration
        func = _get_or_create_function(region=region, instance_type=instance_type)

        # https://modal.com/docs/reference/modal.Client#from_credentials
        client = modal.Client.from_credentials(
            token_id=WEBRTC_MODAL_TOKEN_ID,
            token_secret=WEBRTC_MODAL_TOKEN_SECRET,
        )
        try:
            modal.App.lookup(
                name=WEBRTC_MODAL_APP_NAME, client=client, create_if_missing=False
            )
        except modal.exception.NotFoundError:
            logger.info("Deploying webrtc modal app %s", WEBRTC_MODAL_APP_NAME)
            app.deploy(name=WEBRTC_MODAL_APP_NAME, client=client)
        deployed_func = modal.Function.from_name(
            app_name=WEBRTC_MODAL_APP_NAME, name=func.info.function_name
        )
        deployed_func.hydrate(client=client)

        # https://modal.com/docs/reference/modal.Queue#ephemeral
        with modal.Queue.ephemeral(client=client) as q:
            logger.info(
                "Spawning webrtc modal function %s (region=%s, instance_type=%s) into modal app %s",
                func.info.function_name,
                region,
                instance_type,
                WEBRTC_MODAL_APP_NAME,
            )
            # https://modal.com/docs/reference/modal.Function#spawn
            deployed_func.spawn(
                webrtc_request=webrtc_request,
                q=q,
            )
            answer = WebRTCWorkerResult.model_validate(
                q.get(block=True, timeout=WEBRTC_MODAL_RESPONSE_TIMEOUT)
            )
            return answer
