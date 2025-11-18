import asyncio
import datetime
from pathlib import Path
from typing import Dict, Optional

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
    WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
    WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
    WEBRTC_MODAL_FUNCTION_GPU,
    WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
    WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT,
    WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
    WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
    WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
    WEBRTC_MODAL_IMAGE_NAME,
    WEBRTC_MODAL_IMAGE_TAG,
    WEBRTC_MODAL_RESPONSE_TIMEOUT,
    WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
)
from inference.core.exceptions import RoboflowAPIUnsuccessfulRequestError
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.webrtc import (
    init_rtc_peer_connection_with_loop,
)
from inference.core.version import __version__
from inference.usage_tracking.collector import usage_collector
from inference.usage_tracking.plan_details import WebRTCPlan

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

    decorator_kwargs = {
        "min_containers": WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
        "buffer_containers": WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
        "scaledown_window": WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
        "timeout": WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
        "enable_memory_snapshot": WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
        "max_inputs": WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
        "env": {
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
            "WEBRTC_MODAL_FUNCTION_GPU": WEBRTC_MODAL_FUNCTION_GPU,
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
            "WEBRTC_MODAL_RTSP_PLACEHOLDER": WEBRTC_MODAL_RTSP_PLACEHOLDER,
            "WEBRTC_MODAL_RTSP_PLACEHOLDER_URL": WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
            "ONNXRUNTIME_EXECUTION_PROVIDERS": "[CUDAExecutionProvider,CPUExecutionProvider]",
        },
        "volumes": {MODEL_CACHE_DIR: rfcache_volume},
    }

    class RTCPeerConnectionModal:
        _webrtc_request: Optional[WebRTCWorkerRequest] = modal.parameter(default=None)
        _exec_session_started: Optional[datetime.datetime] = modal.parameter(
            default=None
        )
        _exec_session_stopped: Optional[datetime.datetime] = modal.parameter(
            default=None
        )

        @modal.method()
        def rtc_peer_connection_modal(
            self,
            webrtc_request: WebRTCWorkerRequest,
            q: modal.Queue,
        ):
            logger.info("*** Spawning %s:", self.__class__.__name__)
            logger.info(
                "webrtc_realtime_processing: %s",
                webrtc_request.webrtc_realtime_processing,
            )
            logger.info("stream_output: %s", webrtc_request.stream_output)
            logger.info("data_output: %s", webrtc_request.data_output)
            logger.info("declared_fps: %s", webrtc_request.declared_fps)
            logger.info("rtsp_url: %s", webrtc_request.rtsp_url)
            logger.info("processing_timeout: %s", webrtc_request.processing_timeout)
            logger.info("requested_plan: %s", webrtc_request.requested_plan)
            logger.info("requested_gpu: %s", webrtc_request.requested_gpu)
            logger.info("requested_region: %s", webrtc_request.requested_region)
            logger.info(
                "ICE servers: %s",
                len(
                    webrtc_request.webrtc_config.iceServers
                    if webrtc_request.webrtc_config
                    else []
                ),
            )
            self._webrtc_request = webrtc_request

            def send_answer(obj: WebRTCWorkerResult):
                logger.info("Sending webrtc answer")
                q.put(obj)

            asyncio.run(
                init_rtc_peer_connection_with_loop(
                    webrtc_request=webrtc_request,
                    send_answer=send_answer,
                )
            )

        # https://modal.com/docs/reference/modal.enter
        # Modal usage calculation is relying on no concurrency and no hot instances
        @modal.enter()
        def start(self):
            self._exec_session_started = datetime.datetime.now()

        @modal.exit()
        def stop(self):
            if not self._webrtc_request:
                return
            self._exec_session_stopped = datetime.datetime.now()
            workflow_id = self._webrtc_request.workflow_configuration.workflow_id
            if not workflow_id:
                if self._webrtc_request.workflow_configuration.workflow_specification:
                    workflow_id = usage_collector._calculate_resource_hash(
                        resource_details=self._webrtc_request.workflow_configuration.workflow_specification
                    )
                else:
                    workflow_id = "unknown"

            # requested plan is guaranteed to be set due to validation in spawn_rtc_peer_connection_modal
            webrtc_plan = self._webrtc_request.requested_plan

            usage_collector.record_usage(
                source=workflow_id,
                category="modal",
                api_key=self._webrtc_request.api_key,
                resource_details={"plan": webrtc_plan},
                execution_duration=(
                    self._exec_session_stopped - self._exec_session_started
                ).total_seconds(),
            )
            usage_collector.push_usage_payloads()

    # Modal derives function name from class name
    # https://modal.com/docs/reference/modal.App#cls
    @app.cls(
        **decorator_kwargs,
    )
    class RTCPeerConnectionModalCPU(RTCPeerConnectionModal):
        pass

    @app.cls(
        **{
            **decorator_kwargs,
            "enable_memory_snapshot": False,
            "gpu": WEBRTC_MODAL_FUNCTION_GPU,  # https://modal.com/docs/guide/gpu#specifying-gpu-type
            "experimental_options": {
                "enable_gpu_snapshot": WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT
            },
        }
    )
    class RTCPeerConnectionModalGPU(RTCPeerConnectionModal):
        pass

    def spawn_rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
    ) -> WebRTCWorkerResult:
        webrtc_plans: Optional[Dict[str, WebRTCPlan]] = (
            usage_collector._plan_details.get_webrtc_plans(
                api_key=webrtc_request.api_key
            )
        )
        if webrtc_plans and webrtc_request.requested_plan:
            if webrtc_request.requested_plan not in webrtc_plans:
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Unknown requested plan {webrtc_request.requested_plan}"
                )
            webrtc_request.requested_gpu = webrtc_plans[
                webrtc_request.requested_plan
            ].gpu
        if (
            webrtc_plans
            and not webrtc_request.requested_plan
            and webrtc_request.requested_gpu
        ):
            gpu_to_plan = {v.gpu: k for k, v in webrtc_plans.items()}
            if webrtc_request.requested_gpu not in gpu_to_plan:
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Requested gpu {webrtc_request.requested_gpu} not associated with any plan"
                )
            webrtc_request.requested_plan = gpu_to_plan[webrtc_request.requested_gpu]

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

        if webrtc_request.requested_gpu:
            RTCPeerConnectionModal = RTCPeerConnectionModalGPU
        else:
            RTCPeerConnectionModal = RTCPeerConnectionModalCPU

        # https://modal.com/docs/reference/modal.Cls#from_name
        deployed_cls = modal.Cls.from_name(
            app_name=app.name,
            name=RTCPeerConnectionModal.__name__,
        )
        deployed_cls.hydrate(client=client)
        if webrtc_request.processing_timeout is None:
            webrtc_request.processing_timeout = WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT
            logger.warning("No timeout specified, using max timeout")
        logger.info(
            "Spawning webrtc modal function with timeout %s",
            webrtc_request.processing_timeout,
        )
        # https://modal.com/docs/reference/modal.Cls#with_options
        cls_with_options = deployed_cls.with_options(
            timeout=webrtc_request.processing_timeout,
        )
        if webrtc_request.requested_gpu is not None:
            logger.info(
                "Spawning webrtc modal function with gpu %s",
                webrtc_request.requested_gpu,
            )
            cls_with_options = cls_with_options.with_options(
                gpu=webrtc_request.requested_gpu,
            )
        if webrtc_request.requested_region:
            logger.info(
                "Spawning webrtc modal function with region %s",
                webrtc_request.requested_region,
            )
            cls_with_options = cls_with_options.with_options(
                region=webrtc_request.requested_region,
            )
        rtc_modal_obj: RTCPeerConnectionModal = cls_with_options()
        # https://modal.com/docs/reference/modal.Queue#ephemeral
        with modal.Queue.ephemeral(client=client) as q:
            logger.info(
                "Spawning webrtc modal function from %s into modal app %s",
                RTCPeerConnectionModal.__name__,
                app.name,
            )
            # https://modal.com/docs/reference/modal.Function#spawn
            rtc_modal_obj.rtc_peer_connection_modal.spawn(
                webrtc_request=webrtc_request,
                q=q,
            )
            answer = WebRTCWorkerResult.model_validate(
                q.get(block=True, timeout=WEBRTC_MODAL_RESPONSE_TIMEOUT)
            )
            return answer
