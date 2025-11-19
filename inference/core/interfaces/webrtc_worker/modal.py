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
    PRELOAD_HF_IDS,
    PRELOAD_MODELS,
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
    WEBRTC_MODAL_GCP_SECRET_NAME,
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
from inference.core.managers.base import ModelManager
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.roboflow_api import get_roboflow_workspace
from inference.core.version import __version__
from inference.models.aliases import resolve_roboflow_model_alias
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.usage_tracking.collector import usage_collector
from inference.usage_tracking.plan_details import WebRTCPlan

try:
    import modal
except ImportError:
    modal = None


if modal is not None:
    docker_tag: str = WEBRTC_MODAL_IMAGE_TAG if WEBRTC_MODAL_IMAGE_TAG else __version__
    if WEBRTC_MODAL_GCP_SECRET_NAME:
        # https://modal.com/docs/reference/modal.Secret#from_name
        secret = modal.Secret.from_name(WEBRTC_MODAL_GCP_SECRET_NAME)
        # https://modal.com/docs/reference/modal.Image#from_gcp_artifact_registry
        video_processing_image = modal.Image.from_gcp_artifact_registry(
            f"{WEBRTC_MODAL_IMAGE_NAME}:{docker_tag}",
            secret=secret,
        )
    else:
        video_processing_image = modal.Image.from_registry(
            f"{WEBRTC_MODAL_IMAGE_NAME}:{docker_tag}"
        )
    video_processing_image = video_processing_image.pip_install("modal").entrypoint([])

    # https://modal.com/docs/reference/modal.Volume
    rfcache_volume = modal.Volume.from_name("rfcache", create_if_missing=True)

    # https://modal.com/docs/reference/modal.App
    app = modal.App(
        name=WEBRTC_MODAL_APP_NAME,
        image=video_processing_image,
        tags={"tag": docker_tag},
    )

    decorator_kwargs = {
        "min_containers": WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
        "buffer_containers": WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
        "scaledown_window": WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
        "timeout": WEBRTC_MODAL_FUNCTION_TIME_LIMIT,
        "enable_memory_snapshot": WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
        "max_inputs": WEBRTC_MODAL_FUNCTION_MAX_INPUTS,
        "env": {
            "ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS": str(
                ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS
            ),
            "ALLOW_WORKFLOW_BLOCKS_ACCESSING_ENVIRONMENTAL_VARIABLES": "False",
            "DISABLE_INFERENCE_CACHE": "True",
            "DISABLE_VERSION_CHECK": "True",
            "HF_HOME": Path(MODEL_CACHE_DIR).joinpath("hf_home").as_posix(),
            "INTERNAL_WEIGHTS_URL_SUFFIX": INTERNAL_WEIGHTS_URL_SUFFIX,
            "METRICS_ENABLED": "False",
            "MODAL_TOKEN_ID": MODAL_TOKEN_ID,
            "MODAL_TOKEN_SECRET": MODAL_TOKEN_SECRET,
            "MODAL_WORKSPACE_NAME": MODAL_WORKSPACE_NAME,
            "MODEL_CACHE_DIR": MODEL_CACHE_DIR,
            "MODELS_CACHE_AUTH_CACHE_MAX_SIZE": str(MODELS_CACHE_AUTH_CACHE_MAX_SIZE),
            "MODELS_CACHE_AUTH_CACHE_TTL": str(MODELS_CACHE_AUTH_CACHE_TTL),
            "MODELS_CACHE_AUTH_ENABLED": str(MODELS_CACHE_AUTH_ENABLED),
            "LOG_LEVEL": LOG_LEVEL,
            "ONNXRUNTIME_EXECUTION_PROVIDERS": "[CUDAExecutionProvider,CPUExecutionProvider]",
            "PRELOAD_HF_IDS": ",".join(PRELOAD_HF_IDS) if PRELOAD_HF_IDS else "",
            "PRELOAD_MODELS": ",".join(PRELOAD_MODELS) if PRELOAD_MODELS else "",
            "PROJECT": PROJECT,
            "ROBOFLOW_INTERNAL_SERVICE_NAME": WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
            "ROBOFLOW_INTERNAL_SERVICE_SECRET": ROBOFLOW_INTERNAL_SERVICE_SECRET,
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
            "TELEMETRY_USE_PERSISTENT_QUEUE": "False",
            "WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS": str(
                WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS
            ),
            "WEBRTC_MODAL_FUNCTION_GPU": WEBRTC_MODAL_FUNCTION_GPU,
            "WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS": str(
                WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS
            ),
            "WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW": str(
                WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW
            ),
            "WEBRTC_MODAL_FUNCTION_TIME_LIMIT": str(WEBRTC_MODAL_FUNCTION_TIME_LIMIT),
            "WEBRTC_MODAL_IMAGE_NAME": WEBRTC_MODAL_IMAGE_NAME,
            "WEBRTC_MODAL_IMAGE_TAG": WEBRTC_MODAL_IMAGE_TAG,
            "WEBRTC_MODAL_RTSP_PLACEHOLDER": WEBRTC_MODAL_RTSP_PLACEHOLDER,
            "WEBRTC_MODAL_RTSP_PLACEHOLDER_URL": WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
        },
        "volumes": {MODEL_CACHE_DIR: rfcache_volume},
    }

    class RTCPeerConnectionModal:
        _model_manager: Optional[ModelManager] = modal.parameter(default=None)

        @modal.method()
        def rtc_peer_connection_modal(
            self,
            webrtc_request: WebRTCWorkerRequest,
            q: modal.Queue,
        ):
            logger.info("*** Spawning %s:", self.__class__.__name__)
            logger.info("Inference tag: %s", docker_tag)
            logger.info(
                "Preloaded models: %s",
                (
                    ", ".join(self._model_manager.models().keys())
                    if self._model_manager
                    else ""
                ),
            )
            _exec_session_started = datetime.datetime.now()
            webrtc_request.processing_session_started = _exec_session_started
            logger.info(
                "WebRTC session started at %s", _exec_session_started.isoformat()
            )
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

            def send_answer(obj: WebRTCWorkerResult):
                logger.info("Sending webrtc answer")
                q.put(obj)

            asyncio.run(
                init_rtc_peer_connection_with_loop(
                    webrtc_request=webrtc_request,
                    send_answer=send_answer,
                    model_manager=self._model_manager,
                )
            )
            _exec_session_stopped = datetime.datetime.now()
            logger.info(
                "WebRTC session stopped at %s",
                _exec_session_stopped.isoformat(),
            )
            workflow_id = webrtc_request.workflow_configuration.workflow_id
            if not workflow_id:
                if webrtc_request.workflow_configuration.workflow_specification:
                    workflow_id = usage_collector._calculate_resource_hash(
                        resource_details=webrtc_request.workflow_configuration.workflow_specification
                    )
                else:
                    workflow_id = "unknown"

            # requested plan is guaranteed to be set due to validation in spawn_rtc_peer_connection_modal
            webrtc_plan = webrtc_request.requested_plan

            video_source = "realtime browser stream"
            if webrtc_request.rtsp_url:
                video_source = "rtsp"
            elif not webrtc_request.webrtc_realtime_processing:
                video_source = "buffered browser stream"

            usage_collector.record_usage(
                source=workflow_id,
                category="modal",
                api_key=webrtc_request.api_key,
                resource_details={
                    "plan": webrtc_plan,
                    "billable": True,
                    "video_source": video_source,
                },
                execution_duration=(
                    _exec_session_stopped - _exec_session_started
                ).total_seconds(),
            )
            usage_collector.push_usage_payloads()
            logger.info("Function completed")

        # https://modal.com/docs/reference/modal.enter
        # https://modal.com/docs/guide/memory-snapshot#gpu-memory-snapshot
        @modal.enter(snap=True)
        def start(self):
            # TODO: pre-load models
            logger.info("Starting container")
            logger.info("Preload hf ids: %s", PRELOAD_HF_IDS)
            logger.info("Preload models: %s", PRELOAD_MODELS)
            if PRELOAD_HF_IDS:
                # Kick off pre-loading of models (owlv2 preloading is based on module-level singleton)
                logger.info("Preloading owlv2 base model")
                import inference.models.owlv2.owlv2
            if PRELOAD_MODELS:
                model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
                self._model_manager = ModelManager(model_registry=model_registry)
                for model_id in PRELOAD_MODELS:
                    try:
                        de_aliased_model_id = resolve_roboflow_model_alias(
                            model_id=model_id
                        )
                        logger.info(f"Preloading model: {de_aliased_model_id}")
                        self._model_manager.add_model(
                            model_id=de_aliased_model_id,
                            api_key=None,
                            countinference=False,
                            service_secret=ROBOFLOW_INTERNAL_SERVICE_SECRET,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to preload model %s: %s",
                            model_id,
                            exc,
                        )

        @modal.exit()
        def stop(self):
            logger.info("Stopping container")

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
            app.deploy(name=WEBRTC_MODAL_APP_NAME, client=client, tag=docker_tag)

        workspace_id = webrtc_request.workflow_configuration.workspace_name
        if not workspace_id:
            try:
                workspace_id = get_roboflow_workspace(api_key=webrtc_request.api_key)
                webrtc_request.workflow_configuration.workspace_name = workspace_id
            except Exception:
                pass

        tags = {"tag": docker_tag}
        if workspace_id:
            tags["workspace_id"] = workspace_id

        # TODO: tag function run

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
