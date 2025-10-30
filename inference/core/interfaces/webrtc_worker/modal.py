import asyncio

from inference.core import logger
from inference.core.env import (
    ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,
    API_KEY,
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
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
)
from inference.core.interfaces.webrtc_worker.entities import WebRTCWorkerRequest
from inference.core.interfaces.webrtc_worker.webrtc import (
    init_rtc_peer_connection_with_loop,
)
from inference.core.version import __version__

try:
    import modal
except ImportError:
    modal = None


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

    # https://modal.com/docs/reference/modal.App
    app = modal.App(
        name=WEBRTC_MODAL_APP_NAME,
        image=video_processing_image,
    )

    # https://modal.com/docs/reference/modal.App#function
    @app.function(
        min_containers=WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
        buffer_containers=WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
        scaledown_window=WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
        timeout=WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
        enable_memory_snapshot=WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
        experimental_options=(
            {"enable_gpu_snapshot": True}
            if WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT
            and WEBRTC_MODAL_FUNCTION_GPU
            else {}
        ),
        gpu=WEBRTC_MODAL_FUNCTION_GPU,
        max_inputs=WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
        env={
            "ROBOFLOW_INTERNAL_SERVICE_SECRET": ROBOFLOW_INTERNAL_SERVICE_SECRET,
            "ROBOFLOW_INTERNAL_SERVICE_NAME": WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
            "PROJECT": PROJECT,
            "API_KEY": API_KEY,
            "INTERNAL_WEIGHTS_URL_SUFFIX": INTERNAL_WEIGHTS_URL_SUFFIX,
            "LOG_LEVEL": LOG_LEVEL,
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
            "TELEMETRY_USE_PERSISTENT_QUEUE": "False",
            "WEBRTC_MODAL_FUNCTION_GPU": WEBRTC_MODAL_FUNCTION_GPU,
            "ONNXRUNTIME_EXECUTION_PROVIDERS": (
                "CUDAExecutionProvider"
                if WEBRTC_MODAL_FUNCTION_GPU
                else "CPUExecutionProvider"
            ),
            "DISABLE_INFERENCE_CACHE": "True",
        },
        volumes={MODEL_CACHE_DIR: rfcache_volume},
    )
    def rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
        q: modal.Queue,
    ):
        logger.info("Received webrtc offer")

        def send_answer(obj):
            logger.info("Sending webrtc answer")
            q.put(obj)

        asyncio.run(
            init_rtc_peer_connection_with_loop(
                webrtc_request=webrtc_request,
                send_answer=send_answer,
            )
        )

    def spawn_rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
    ):
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
            app_name=app.name, name=rtc_peer_connection_modal.info.function_name
        )
        deployed_func.hydrate(client=client)
        # https://modal.com/docs/reference/modal.Queue#ephemeral
        with modal.Queue.ephemeral(client=client) as q:
            logger.info(
                "Spawning webrtc modal function %s into modal app %s",
                rtc_peer_connection_modal.info.function_name,
                app.name,
            )
            # https://modal.com/docs/reference/modal.Function#spawn
            deployed_func.spawn(
                webrtc_request=webrtc_request,
                q=q,
            )
            answer = q.get(block=True, timeout=WEBRTC_MODAL_RESPONSE_TIMEOUT)
            return None, answer
