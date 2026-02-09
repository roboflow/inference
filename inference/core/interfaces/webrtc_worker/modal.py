import asyncio
import datetime
import os
import subprocess
import time
from pathlib import Path
from queue import Empty
from typing import Callable, Dict, Optional

_module_import_start = time.perf_counter()

from inference.core import logger

logger.warning("[COLD_START] Core logger imported in %.3fs", time.perf_counter() - _module_import_start)

_env_import_start = time.perf_counter()
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
    WEBRTC_DATA_CHANNEL_ACK_WINDOW,
    WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT,
    WEBRTC_GZIP_PREVIEW_FRAME_COMPRESSION,
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
    WEBRTC_MODAL_MIN_CPU_CORES,
    WEBRTC_MODAL_MIN_RAM_MB,
    WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
    WEBRTC_MODAL_PRELOAD_HF_IDS,
    WEBRTC_MODAL_PRELOAD_MODELS,
    WEBRTC_MODAL_RESPONSE_TIMEOUT,
    WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
    WEBRTC_MODAL_SHUTDOWN_RESERVE,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WEBRTC_MODAL_USAGE_QUOTA_ENABLED,
    WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
)
logger.warning("[COLD_START] inference.core.env imported in %.3fs", time.perf_counter() - _env_import_start)

_exceptions_import_start = time.perf_counter()
from inference.core.exceptions import (
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
logger.warning("[COLD_START] inference.core.exceptions imported in %.3fs", time.perf_counter() - _exceptions_import_start)

_entities_import_start = time.perf_counter()
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
logger.warning("[COLD_START] webrtc_worker.entities imported in %.3fs", time.perf_counter() - _entities_import_start)

_utils_import_start = time.perf_counter()
from inference.core.interfaces.webrtc_worker.utils import (
    warmup_cuda,
    workflow_contains_instant_model,
    workflow_contains_preloaded_model,
)
logger.warning("[COLD_START] webrtc_worker.utils imported in %.3fs", time.perf_counter() - _utils_import_start)

_watchdog_import_start = time.perf_counter()
from inference.core.interfaces.webrtc_worker.watchdog import Watchdog
logger.warning("[COLD_START] webrtc_worker.watchdog imported in %.3fs", time.perf_counter() - _watchdog_import_start)

_model_manager_import_start = time.perf_counter()
from inference.core.managers.base import ModelManager
logger.warning("[COLD_START] managers.base.ModelManager imported in %.3fs", time.perf_counter() - _model_manager_import_start)

_registry_import_start = time.perf_counter()
from inference.core.registries.roboflow import RoboflowModelRegistry
logger.warning("[COLD_START] registries.roboflow imported in %.3fs", time.perf_counter() - _registry_import_start)

_roboflow_api_import_start = time.perf_counter()
from inference.core.roboflow_api import (
    get_roboflow_workspace,
    get_workflow_specification,
)
logger.warning("[COLD_START] roboflow_api imported in %.3fs", time.perf_counter() - _roboflow_api_import_start)

_version_import_start = time.perf_counter()
from inference.core.version import __version__
logger.warning("[COLD_START] core.version imported in %.3fs", time.perf_counter() - _version_import_start)

_aliases_import_start = time.perf_counter()
from inference.models.aliases import resolve_roboflow_model_alias
logger.warning("[COLD_START] models.aliases imported in %.3fs", time.perf_counter() - _aliases_import_start)

_owlv2_import_start = time.perf_counter()
from inference.models.owlv2.owlv2 import PRELOADED_HF_MODELS, preload_owlv2_model
logger.warning("[COLD_START] models.owlv2 imported in %.3fs", time.perf_counter() - _owlv2_import_start)

_model_types_import_start = time.perf_counter()
from inference.models.utils import ROBOFLOW_MODEL_TYPES
logger.warning("[COLD_START] models.utils imported in %.3fs", time.perf_counter() - _model_types_import_start)

_usage_collector_import_start = time.perf_counter()
from inference.usage_tracking.collector import usage_collector
logger.warning("[COLD_START] usage_tracking.collector imported in %.3fs", time.perf_counter() - _usage_collector_import_start)

_plan_details_import_start = time.perf_counter()
from inference.usage_tracking.plan_details import WebRTCPlan
logger.warning("[COLD_START] usage_tracking.plan_details imported in %.3fs", time.perf_counter() - _plan_details_import_start)

_modal_import_start = time.perf_counter()
try:
    import modal
    logger.warning("[COLD_START] modal library imported in %.3fs", time.perf_counter() - _modal_import_start)
except ImportError:
    modal = None
    logger.warning("[COLD_START] modal library import failed in %.3fs", time.perf_counter() - _modal_import_start)

logger.warning("[COLD_START] Total module imports completed in %.3fs", time.perf_counter() - _module_import_start)


# https://modal.com/docs/guide/environment_variables#environment-variables
MODAL_CLOUD_PROVIDER = os.getenv("MODAL_CLOUD_PROVIDER")
MODAL_IMAGE_ID = os.getenv("MODAL_IMAGE_ID")
MODAL_REGION = os.getenv("MODAL_REGION")
MODAL_TASK_ID = os.getenv("MODAL_TASK_ID")
MODAL_ENVIRONMENT = os.getenv("MODAL_ENVIRONMENT")
MODAL_IDENTITY_TOKEN = os.getenv("MODAL_IDENTITY_TOKEN")


def check_nvidia_smi_gpu() -> str:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        return gpu
    except subprocess.CalledProcessError:
        return ""


if modal is not None:
    _modal_setup_start = time.perf_counter()
    logger.warning("[COLD_START] Starting Modal app setup...")
    
    docker_tag: str = WEBRTC_MODAL_IMAGE_TAG if WEBRTC_MODAL_IMAGE_TAG else __version__
    logger.warning("[COLD_START] Using docker tag: %s", docker_tag)
    
    _image_setup_start = time.perf_counter()
    if WEBRTC_MODAL_GCP_SECRET_NAME:
        # https://modal.com/docs/reference/modal.Secret#from_name
        logger.warning("[COLD_START] Loading GCP secret: %s", WEBRTC_MODAL_GCP_SECRET_NAME)
        secret = modal.Secret.from_name(WEBRTC_MODAL_GCP_SECRET_NAME)
        # https://modal.com/docs/reference/modal.Image#from_gcp_artifact_registry
        logger.warning("[COLD_START] Creating image from GCP artifact registry: %s:%s", WEBRTC_MODAL_IMAGE_NAME, docker_tag)
        video_processing_image = modal.Image.from_gcp_artifact_registry(
            f"{WEBRTC_MODAL_IMAGE_NAME}:{docker_tag}",
            secret=secret,
        )
    else:
        logger.warning("[COLD_START] Creating image from registry: %s:%s", WEBRTC_MODAL_IMAGE_NAME, docker_tag)
        video_processing_image = modal.Image.from_registry(
            f"{WEBRTC_MODAL_IMAGE_NAME}:{docker_tag}"
        )
    logger.warning("[COLD_START] Base image defined in %.3fs", time.perf_counter() - _image_setup_start)

    _image_customize_start = time.perf_counter()
    video_processing_image = (
        video_processing_image.apt_install("ffmpeg").pip_install("modal").entrypoint([])
    )
    logger.warning("[COLD_START] Image customization (apt+pip) defined in %.3fs", time.perf_counter() - _image_customize_start)

    # https://modal.com/docs/reference/modal.Volume
    _volume_setup_start = time.perf_counter()
    rfcache_volume = modal.Volume.from_name("rfcache", create_if_missing=True)
    logger.warning("[COLD_START] Volume 'rfcache' referenced in %.3fs", time.perf_counter() - _volume_setup_start)

    # https://modal.com/docs/reference/modal.App
    _app_create_start = time.perf_counter()
    app = modal.App(
        name=WEBRTC_MODAL_APP_NAME,
        image=video_processing_image,
        tags={"tag": docker_tag},
    )
    logger.warning("[COLD_START] Modal App '%s' created in %.3fs", WEBRTC_MODAL_APP_NAME, time.perf_counter() - _app_create_start)
    logger.warning("[COLD_START] Modal app setup completed in %.3fs", time.perf_counter() - _modal_setup_start)

    decorator_kwargs = {
        "min_containers": WEBRTC_MODAL_FUNCTION_MIN_CONTAINERS,
        "buffer_containers": WEBRTC_MODAL_FUNCTION_BUFFER_CONTAINERS,
        "scaledown_window": WEBRTC_MODAL_FUNCTION_SCALEDOWN_WINDOW,
        "memory": WEBRTC_MODAL_MIN_RAM_MB,
        "cpu": WEBRTC_MODAL_MIN_CPU_CORES,
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
            "PROJECT": PROJECT,
            "PYTHONASYNCIODEBUG": str(os.getenv("PYTHONASYNCIODEBUG", "0")),
            "ROBOFLOW_INTERNAL_SERVICE_NAME": WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
            "ROBOFLOW_INTERNAL_SERVICE_SECRET": ROBOFLOW_INTERNAL_SERVICE_SECRET,
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
            "TELEMETRY_USE_PERSISTENT_QUEUE": "False",
            "TELEMETRY_API_PLAN_CACHE_TTL_SECONDS": str(
                os.getenv("TELEMETRY_API_PLAN_CACHE_TTL_SECONDS", 60)
            ),
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
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
            "WEBRTC_MODAL_MIN_CPU_CORES": str(
                WEBRTC_MODAL_MIN_CPU_CORES if WEBRTC_MODAL_MIN_CPU_CORES else ""
            ),
            "WEBRTC_MODAL_MIN_RAM_MB": str(
                WEBRTC_MODAL_MIN_RAM_MB if WEBRTC_MODAL_MIN_RAM_MB else ""
            ),
            "WEBRTC_MODAL_MODELS_PRELOAD_API_KEY": (
                str(WEBRTC_MODAL_MODELS_PRELOAD_API_KEY)
                if WEBRTC_MODAL_MODELS_PRELOAD_API_KEY
                else ""
            ),
            "WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT": str(
                WEBRTC_DATA_CHANNEL_BUFFER_SIZE_LIMIT
            ),
            "WEBRTC_DATA_CHANNEL_ACK_WINDOW": str(WEBRTC_DATA_CHANNEL_ACK_WINDOW),
            "WEBRTC_MODAL_RTSP_PLACEHOLDER": WEBRTC_MODAL_RTSP_PLACEHOLDER,
            "WEBRTC_MODAL_RTSP_PLACEHOLDER_URL": WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
            "WEBRTC_MODAL_SHUTDOWN_RESERVE": str(WEBRTC_MODAL_SHUTDOWN_RESERVE),
            "WEBRTC_MODAL_USAGE_QUOTA_ENABLED": str(WEBRTC_MODAL_USAGE_QUOTA_ENABLED),
            "WEBRTC_MODAL_WATCHDOG_TIMEMOUT": str(WEBRTC_MODAL_WATCHDOG_TIMEMOUT),
            "WEBRTC_GZIP_PREVIEW_FRAME_COMPRESSION": str(
                WEBRTC_GZIP_PREVIEW_FRAME_COMPRESSION
            ),
        },
        "volumes": {MODEL_CACHE_DIR: rfcache_volume},
    }

    async def run_rtc_peer_connection_with_watchdog(
        webrtc_request: WebRTCWorkerRequest,
        send_answer: Callable[[WebRTCWorkerResult], None],
        model_manager: ModelManager,
        watchdog: Watchdog,
    ):
        _webrtc_import_start = time.perf_counter()
        logger.warning("[COLD_START] Importing webrtc module...")
        from inference.core.interfaces.webrtc_worker.webrtc import (
            init_rtc_peer_connection_with_loop,
        )
        logger.warning("[COLD_START] webrtc module imported in %.3fs", time.perf_counter() - _webrtc_import_start)

        logger.warning("[COLD_START] Creating RTC peer connection task...")
        rtc_peer_connection_task = asyncio.create_task(
            init_rtc_peer_connection_with_loop(
                webrtc_request=webrtc_request,
                send_answer=send_answer,
                model_manager=model_manager,
                heartbeat_callback=watchdog.heartbeat,
            )
        )

        loop = asyncio.get_running_loop()

        def on_timeout(message: Optional[str] = ""):
            msg = "Cancelled by watchdog"
            if message:
                msg += f": {message}"
            # Use call_soon_threadsafe since this callback is invoked from the watchdog thread
            loop.call_soon_threadsafe(rtc_peer_connection_task.cancel, msg)

        watchdog.on_timeout = on_timeout
        watchdog.start()

        try:
            await rtc_peer_connection_task
            logger.info("Task completed uninterrupted")
        except modal.exception.InputCancellation:
            logger.warning("Modal function was cancelled")
        except asyncio.CancelledError as exc:
            logger.warning("WebRTC connection task was cancelled (%s)", exc)
        except Exception as exc:
            logger.error(exc)
        finally:
            watchdog.stop()

    class RTCPeerConnectionModal:
        _model_manager: Optional[ModelManager] = modal.parameter(
            default=None, init=False
        )
        _gpu: Optional[str] = modal.parameter(default=None, init=False)
        _container_startup_time_seconds: Optional[float] = modal.parameter(
            default=0, init=False
        )
        _function_call_number_on_container: Optional[int] = modal.parameter(
            default=0, init=False
        )
        _cold_start: Optional[bool] = modal.parameter(default=True, init=False)

        @modal.method()
        def rtc_peer_connection_modal(
            self,
            webrtc_request: WebRTCWorkerRequest,
            q: modal.Queue,
        ):
            _method_start = time.perf_counter()
            logger.info("[COLD_START] ========== MODAL METHOD EXECUTION BEGIN ==========")
            logger.info("[COLD_START] Container received function call, starting execution...")
            
            _workspace_api_start = time.perf_counter()
            logger.warning("[COLD_START] Fetching workspace ID from API...")
            _workspace_id = get_roboflow_workspace(api_key=webrtc_request.api_key)
            logger.warning("[COLD_START] Workspace ID fetched in %.3fs: %s", time.perf_counter() - _workspace_api_start, _workspace_id)

            workflow_id = webrtc_request.workflow_configuration.workflow_id
            if not workflow_id:
                if webrtc_request.workflow_configuration.workflow_specification:
                    workflow_id = usage_collector._calculate_resource_hash(
                        resource_details=webrtc_request.workflow_configuration.workflow_specification
                    )
                else:
                    workflow_id = "unknown"

            self._function_call_number_on_container += 1
            logger.info("*** Spawning %s:", self.__class__.__name__)
            logger.info("Running on %s", self._gpu)
            logger.info("Inference tag: %s", docker_tag)
            logger.info("Workspace ID: %s", _workspace_id)
            logger.info("Workflow ID: %s", workflow_id)
            logger.info(
                "Preloaded models: %s",
                (
                    ", ".join(self._model_manager.models().keys())
                    if self._model_manager
                    else ""
                ),
            )
            logger.info(
                "Preloaded hf models: %s", ", ".join(PRELOADED_HF_MODELS.keys())
            )
            logger.info("Cold start: %s", self._cold_start)
            logger.info(
                "Function call number on container: %s",
                self._function_call_number_on_container,
            )
            logger.info(
                "Container startup time: %s", self._container_startup_time_seconds
            )
            _exec_session_started = datetime.datetime.now()
            webrtc_request.processing_session_started = _exec_session_started
            # Modal cancels based on time taken during entry hook
            if self._function_call_number_on_container == 1 and self._cold_start:
                logger.info(
                    "Subtracting container startup time (%s) from processing session started (%s)",
                    self._container_startup_time_seconds,
                    webrtc_request.processing_session_started,
                )
                webrtc_request.processing_session_started -= datetime.timedelta(
                    seconds=self._container_startup_time_seconds
                )
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
            logger.info("watchdog_timeout: %s", WEBRTC_MODAL_WATCHDOG_TIMEMOUT)
            logger.info("requested_plan: %s", webrtc_request.requested_plan)
            logger.info("requested_region: %s", webrtc_request.requested_region)
            logger.info(
                "ICE servers: %s",
                len(
                    webrtc_request.webrtc_config.iceServers
                    if webrtc_request.webrtc_config
                    else []
                ),
            )
            logger.info(
                "WEBRTC_MODAL_MIN_CPU_CORES: %s",
                WEBRTC_MODAL_MIN_CPU_CORES or "not set",
            )
            logger.info(
                "WEBRTC_MODAL_MIN_RAM_MB: %s", WEBRTC_MODAL_MIN_RAM_MB or "not set"
            )
            logger.info("MODAL_CLOUD_PROVIDER: %s", MODAL_CLOUD_PROVIDER)
            logger.info("MODAL_IMAGE_ID: %s", MODAL_IMAGE_ID)
            logger.info("MODAL_REGION: %s", MODAL_REGION)
            logger.info("MODAL_TASK_ID: %s", MODAL_TASK_ID)
            logger.info("MODAL_ENVIRONMENT: %s", MODAL_ENVIRONMENT)
            logger.info("MODAL_IDENTITY_TOKEN: %s", MODAL_IDENTITY_TOKEN)
            logger.info("[COLD_START] Method setup and logging completed in %.3fs", time.perf_counter() - _method_start)

            def send_answer(obj: WebRTCWorkerResult):
                logger.info("Sending webrtc answer")
                if obj.error_message:
                    logger.error(
                        "Error: %s (%s)", obj.error_message, obj.exception_type
                    )
                # Queue with no limit, below will never block
                q.put(obj)

            if webrtc_request.processing_timeout == 0:
                error_msg = "Processing timeout is 0, skipping processing"
                logger.info(error_msg)
                send_answer(WebRTCWorkerResult(error_message=error_msg))
                return
            if (
                not webrtc_request.webrtc_offer
                or not webrtc_request.webrtc_offer.sdp
                or not webrtc_request.webrtc_offer.type
            ):
                error_msg = "Webrtc offer is missing, skipping processing"
                logger.info(error_msg)
                send_answer(WebRTCWorkerResult(error_message=error_msg))
                return

            _watchdog_create_start = time.perf_counter()
            logger.warning("[COLD_START] Creating watchdog...")
            watchdog = Watchdog(
                api_key=webrtc_request.api_key,
                timeout_seconds=WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
            )
            logger.warning("[COLD_START] Watchdog created in %.3fs", time.perf_counter() - _watchdog_create_start)

            try:
                _asyncio_run_start = time.perf_counter()
                logger.info("[COLD_START] Starting asyncio.run for WebRTC connection...")
                asyncio.run(
                    run_rtc_peer_connection_with_watchdog(
                        webrtc_request=webrtc_request,
                        send_answer=send_answer,
                        model_manager=self._model_manager,
                        watchdog=watchdog,
                    )
                )
                logger.info("[COLD_START] asyncio.run completed in %.3fs", time.perf_counter() - _asyncio_run_start)
            except modal.exception.InputCancellation:
                logger.warning("Modal function was cancelled")
            except asyncio.CancelledError as exc:
                logger.warning("WebRTC connection task was cancelled (%s)", exc)
            except Exception as exc:
                logger.warning("Unhandled exception: %s", exc)
            finally:
                watchdog.stop()

            _exec_session_stopped = datetime.datetime.now()
            logger.info(
                "WebRTC session stopped at %s",
                _exec_session_stopped.isoformat(),
            )
            if watchdog.total_heartbeats == 0:
                raise Exception(
                    "WebRTC worker was terminated before processing a single frame"
                )

            # requested plan is guaranteed to be set due to validation in spawn_rtc_peer_connection_modal
            webrtc_plan = webrtc_request.requested_plan

            video_source = "realtime browser stream"
            if webrtc_request.rtsp_url:
                video_source = "rtsp"
            elif not webrtc_request.webrtc_realtime_processing:
                video_source = "buffered browser stream"
            else:
                video_source = "realtime browser stream"

            usage_collector.record_usage(
                source=workflow_id,
                category="modal",
                api_key=webrtc_request.api_key,
                resource_details={
                    "plan": webrtc_plan,
                    "billable": True,
                    "video_source": video_source,
                    "is_preview": webrtc_request.is_preview,
                },
                execution_duration=(
                    _exec_session_stopped - _exec_session_started
                ).total_seconds(),
            )
            usage_collector.push_usage_payloads()
            logger.info("Function completed")

        @modal.exit()
        def stop(self):
            logger.info("Stopping container")

    # Modal derives function name from class name
    # https://modal.com/docs/reference/modal.App#cls
    @app.cls(
        **decorator_kwargs,
    )
    class RTCPeerConnectionModalCPU(RTCPeerConnectionModal):
        # https://modal.com/docs/guide/parametrized-functions#parametrized-functions
        preload_models: str = modal.parameter(default="")
        preload_hf_ids: str = modal.parameter(default="")

        # https://modal.com/docs/reference/modal.enter
        @modal.enter(snap=True)
        def start(self):
            _cpu_start_time = time.perf_counter()
            logger.info("[COLD_START] ========== CPU CONTAINER STARTUP BEGIN ==========")
            # TODO: pre-load models on CPU
            logger.info("[COLD_START] Starting CPU container")
            self._gpu = "CPU"
            self._cold_start = False
            logger.info("[COLD_START] CPU container startup completed in %.3fs", time.perf_counter() - _cpu_start_time)
            logger.info("[COLD_START] ========== CPU CONTAINER STARTUP END ==========")

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
        # https://modal.com/docs/guide/parametrized-functions#parametrized-functions
        preload_models: str = modal.parameter(default="")
        preload_hf_ids: str = modal.parameter(default="")

        # https://modal.com/docs/reference/modal.enter
        # https://modal.com/docs/guide/memory-snapshot#gpu-memory-snapshot
        @modal.enter(snap=True)
        def start(self):
            _gpu_start_time = time.perf_counter()
            logger.info("[COLD_START] ========== GPU CONTAINER STARTUP BEGIN ==========")
            
            self._cold_start = False
            time_start = time.time()
            
            # CUDA warmup
            _cuda_warmup_start = time.perf_counter()
            logger.info("[COLD_START] Starting CUDA warmup...")
            warmup_cuda(max_retries=10, retry_delay=0.5)
            logger.info("[COLD_START] CUDA warmup completed in %.3fs", time.perf_counter() - _cuda_warmup_start)
            
            # GPU detection
            _gpu_detect_start = time.perf_counter()
            logger.info("[COLD_START] Detecting GPU via nvidia-smi...")
            self._gpu = check_nvidia_smi_gpu()
            logger.info("[COLD_START] GPU detection completed in %.3fs: %s", time.perf_counter() - _gpu_detect_start, self._gpu)
            
            logger.info("[COLD_START] Starting GPU container on %s", self._gpu)
            logger.info("[COLD_START] Preload hf ids: %s", self.preload_hf_ids)
            logger.info("[COLD_START] Preload models: %s", self.preload_models)
            
            # HF model preloading (OwlV2)
            if self.preload_hf_ids:
                _hf_preload_total_start = time.perf_counter()
                logger.info("[COLD_START] Starting HuggingFace model preloading...")
                preload_hf_ids = [m.strip() for m in self.preload_hf_ids.split(",")]
                for preload_hf_id in preload_hf_ids:
                    _hf_model_start = time.perf_counter()
                    logger.info("[COLD_START] Preloading owlv2 base model: %s", preload_hf_id)
                    preload_owlv2_model(preload_hf_id)
                    logger.info("[COLD_START] Preloaded owlv2 model '%s' in %.3fs", preload_hf_id, time.perf_counter() - _hf_model_start)
                logger.info("[COLD_START] All HuggingFace models preloaded in %.3fs", time.perf_counter() - _hf_preload_total_start)
            else:
                logger.info("[COLD_START] No HuggingFace models to preload")
            
            # Roboflow model preloading
            if self.preload_models:
                _rf_preload_total_start = time.perf_counter()
                logger.info("[COLD_START] Starting Roboflow model preloading...")
                preload_models = []
                if self.preload_models:
                    preload_models = [m.strip() for m in self.preload_models.split(",")]
                
                _registry_create_start = time.perf_counter()
                logger.info("[COLD_START] Creating RoboflowModelRegistry...")
                model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
                logger.info("[COLD_START] RoboflowModelRegistry created in %.3fs", time.perf_counter() - _registry_create_start)
                
                _manager_create_start = time.perf_counter()
                logger.info("[COLD_START] Creating ModelManager...")
                model_manager = ModelManager(model_registry=model_registry)
                logger.info("[COLD_START] ModelManager created in %.3fs", time.perf_counter() - _manager_create_start)
                
                for model_id in preload_models:
                    try:
                        _alias_resolve_start = time.perf_counter()
                        de_aliased_model_id = resolve_roboflow_model_alias(
                            model_id=model_id
                        )
                        logger.info("[COLD_START] Resolved model alias '%s' -> '%s' in %.3fs", model_id, de_aliased_model_id, time.perf_counter() - _alias_resolve_start)
                        
                        _model_add_start = time.perf_counter()
                        logger.info("[COLD_START] Adding model to manager: %s", de_aliased_model_id)
                        model_manager.add_model(
                            model_id=de_aliased_model_id,
                            api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
                            countinference=False,
                            service_secret=ROBOFLOW_INTERNAL_SERVICE_SECRET,
                        )
                        logger.info("[COLD_START] Model '%s' added and loaded in %.3fs", de_aliased_model_id, time.perf_counter() - _model_add_start)
                    except Exception as exc:
                        logger.error(
                            "[COLD_START] Failed to preload model %s: %s",
                            model_id,
                            exc,
                        )
                self._model_manager = model_manager
                logger.info("[COLD_START] All Roboflow models preloaded in %.3fs", time.perf_counter() - _rf_preload_total_start)
            else:
                logger.info("[COLD_START] No Roboflow models to preload")
            
            time_end = time.time()
            self._container_startup_time_seconds = time_end - time_start
            
            logger.info("[COLD_START] GPU container startup completed in %.3fs", time.perf_counter() - _gpu_start_time)
            logger.info("[COLD_START] ========== GPU CONTAINER STARTUP END ==========")

    def spawn_rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
    ) -> WebRTCWorkerResult:
        _spawn_total_start = time.perf_counter()
        logger.info("[COLD_START] ========== SPAWN RTC PEER CONNECTION BEGIN ==========")
        
        requested_gpu: Optional[str] = None
        requested_ram_mb: Optional[int] = None
        requested_cpu_cores: Optional[int] = None
        
        _plan_lookup_start = time.perf_counter()
        logger.warning("[COLD_START] Looking up WebRTC plans...")
        webrtc_plans: Optional[Dict[str, WebRTCPlan]] = (
            usage_collector._plan_details.get_webrtc_plans(
                api_key=webrtc_request.api_key
            )
        )
        logger.warning("[COLD_START] WebRTC plans lookup completed in %.3fs, found %d plans", time.perf_counter() - _plan_lookup_start, len(webrtc_plans) if webrtc_plans else 0)
        
        if webrtc_plans and webrtc_request.requested_plan:
            if webrtc_request.requested_plan not in webrtc_plans:
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Unknown requested plan {webrtc_request.requested_plan}, available plans: {', '.join(webrtc_plans.keys())}"
                )
            requested_gpu = webrtc_plans[webrtc_request.requested_plan].gpu
            requested_ram_mb = webrtc_plans[webrtc_request.requested_plan].ram_mb
            requested_cpu_cores = webrtc_plans[webrtc_request.requested_plan].cpu_cores
            logger.warning("[COLD_START] Using requested plan: %s (gpu=%s, ram=%s, cpu=%s)", webrtc_request.requested_plan, requested_gpu, requested_ram_mb, requested_cpu_cores)

        # TODO: requested_gpu is replaced with requested_plan
        if (
            webrtc_plans
            and not webrtc_request.requested_plan
            and webrtc_request.requested_gpu
        ):
            gpu_to_plan = {v.gpu: k for k, v in webrtc_plans.items()}
            if webrtc_request.requested_gpu not in gpu_to_plan:
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Requested gpu {webrtc_request.requested_gpu} not associated with any plan, available gpus: {', '.join(gpu_to_plan.keys())}"
                )
            webrtc_request.requested_plan = gpu_to_plan[webrtc_request.requested_gpu]
            requested_gpu = webrtc_plans[webrtc_request.requested_plan].gpu

        # https://modal.com/docs/reference/modal.Client#from_credentials
        _client_create_start = time.perf_counter()
        logger.warning("[COLD_START] Creating Modal client from credentials...")
        client = modal.Client.from_credentials(
            token_id=WEBRTC_MODAL_TOKEN_ID,
            token_secret=WEBRTC_MODAL_TOKEN_SECRET,
        )
        logger.warning("[COLD_START] Modal client created in %.3fs", time.perf_counter() - _client_create_start)
        
        _app_lookup_start = time.perf_counter()
        logger.warning("[COLD_START] Looking up Modal app '%s'...", WEBRTC_MODAL_APP_NAME)
        try:
            modal.App.lookup(
                name=WEBRTC_MODAL_APP_NAME, client=client, create_if_missing=False
            )
            logger.warning("[COLD_START] Modal app lookup completed in %.3fs (app exists)", time.perf_counter() - _app_lookup_start)
        except modal.exception.NotFoundError:
            logger.info("[COLD_START] Deploying webrtc modal app %s (app not found)", WEBRTC_MODAL_APP_NAME)
            _app_deploy_start = time.perf_counter()
            app.deploy(name=WEBRTC_MODAL_APP_NAME, client=client, tag=docker_tag)
            logger.info("[COLD_START] Modal app deployed in %.3fs", time.perf_counter() - _app_deploy_start)

        _workspace_lookup_start = time.perf_counter()
        logger.warning("[COLD_START] Getting workspace and workflow info...")
        workspace_id = webrtc_request.workflow_configuration.workspace_name
        if not workspace_id:
            logger.warning("[COLD_START] Fetching workspace ID from API...")
            workspace_id = get_roboflow_workspace(api_key=webrtc_request.api_key)
            webrtc_request.workflow_configuration.workspace_name = workspace_id
            logger.warning("[COLD_START] Workspace ID fetched: %s", workspace_id)
        if not webrtc_request.workflow_configuration.workflow_specification:
            _workflow_spec_start = time.perf_counter()
            logger.warning("[COLD_START] Fetching workflow specification from API...")
            webrtc_request.workflow_configuration.workflow_specification = (
                get_workflow_specification(
                    api_key=webrtc_request.api_key,
                    workspace_id=webrtc_request.workflow_configuration.workspace_name,
                    workflow_id=webrtc_request.workflow_configuration.workflow_id,
                )
            )
            logger.warning("[COLD_START] Workflow specification fetched in %.3fs", time.perf_counter() - _workflow_spec_start)
        logger.warning("[COLD_START] Workspace and workflow info completed in %.3fs", time.perf_counter() - _workspace_lookup_start)
        
        tags = {"tag": docker_tag}
        if workspace_id:
            tags["workspace_id"] = workspace_id

        # TODO: tag function run

        # Modal parametrization
        _parametrization_start = time.perf_counter()
        logger.warning("[COLD_START] Checking model parametrization...")
        preload_hf_ids = ""
        if WEBRTC_MODAL_PRELOAD_HF_IDS and workflow_contains_instant_model(
            workflow_specification=webrtc_request.workflow_configuration.workflow_specification
        ):
            logger.info("[COLD_START] Parametrized preload hf ids: %s", WEBRTC_MODAL_PRELOAD_HF_IDS)
            preload_hf_ids = WEBRTC_MODAL_PRELOAD_HF_IDS

        preload_models = ""
        if WEBRTC_MODAL_PRELOAD_MODELS and workflow_contains_preloaded_model(
            workflow_specification=webrtc_request.workflow_configuration.workflow_specification,
            preload_models=[m.strip() for m in WEBRTC_MODAL_PRELOAD_MODELS.split(",")],
        ):
            logger.info("[COLD_START] Parametrized preload models: %s", WEBRTC_MODAL_PRELOAD_MODELS)
            preload_models = WEBRTC_MODAL_PRELOAD_MODELS
        logger.warning("[COLD_START] Model parametrization completed in %.3fs", time.perf_counter() - _parametrization_start)

        if requested_gpu:
            RTCPeerConnectionModal = RTCPeerConnectionModalGPU
            logger.warning("[COLD_START] Using GPU container class: RTCPeerConnectionModalGPU")
        else:
            RTCPeerConnectionModal = RTCPeerConnectionModalCPU
            logger.warning("[COLD_START] Using CPU container class: RTCPeerConnectionModalCPU")

        # https://modal.com/docs/reference/modal.Cls#from_name
        _cls_lookup_start = time.perf_counter()
        logger.warning("[COLD_START] Looking up Modal class '%s' from app '%s'...", RTCPeerConnectionModal.__name__, app.name)
        deployed_cls = modal.Cls.from_name(
            app_name=app.name,
            name=RTCPeerConnectionModal.__name__,
        )
        logger.warning("[COLD_START] Modal class looked up in %.3fs", time.perf_counter() - _cls_lookup_start)
        
        _cls_hydrate_start = time.perf_counter()
        logger.warning("[COLD_START] Hydrating Modal class...")
        deployed_cls.hydrate(client=client)
        logger.warning("[COLD_START] Modal class hydrated in %.3fs", time.perf_counter() - _cls_hydrate_start)
        
        if webrtc_request.processing_timeout is None:
            webrtc_request.processing_timeout = WEBRTC_MODAL_FUNCTION_MAX_TIME_LIMIT
            logger.warning("[COLD_START] No timeout specified, using max timeout")
        logger.info(
            "[COLD_START] Spawning webrtc modal function with timeout %s",
            webrtc_request.processing_timeout,
        )
        
        # https://modal.com/docs/reference/modal.Cls#with_options
        _options_start = time.perf_counter()
        logger.warning("[COLD_START] Applying container options...")
        cls_with_options = deployed_cls.with_options(
            timeout=webrtc_request.processing_timeout,
        )
        if requested_gpu is not None:
            logger.info(
                "[COLD_START] Spawning webrtc modal function with gpu %s",
                requested_gpu,
            )
            # Specify fallback GPU
            # TODO: with_options does not support gpu fallback
            # https://modal.com/docs/examples/gpu_fallbacks#set-fallback-gpus
            cls_with_options = cls_with_options.with_options(
                gpu=requested_gpu,
            )
        if webrtc_request.requested_region:
            logger.info(
                "[COLD_START] Spawning webrtc modal function with region %s",
                webrtc_request.requested_region,
            )
            cls_with_options = cls_with_options.with_options(
                region=webrtc_request.requested_region,
            )
        if requested_ram_mb is not None:
            logger.info(
                "[COLD_START] Spawning webrtc modal function with ram %s",
                requested_ram_mb,
            )
            cls_with_options = cls_with_options.with_options(
                memory=requested_ram_mb,
            )
        if requested_cpu_cores is not None:
            logger.info(
                "[COLD_START] Spawning webrtc modal function with cpu cores %s",
                requested_cpu_cores,
            )
            cls_with_options = cls_with_options.with_options(
                cpu=requested_cpu_cores,
            )
        logger.warning("[COLD_START] Container options applied in %.3fs", time.perf_counter() - _options_start)
        
        _obj_create_start = time.perf_counter()
        logger.warning("[COLD_START] Creating Modal class instance with params (preload_hf_ids=%s, preload_models=%s)...", preload_hf_ids, preload_models)
        rtc_modal_obj: RTCPeerConnectionModal = cls_with_options(
            preload_hf_ids=preload_hf_ids,
            preload_models=preload_models,
        )
        logger.warning("[COLD_START] Modal class instance created in %.3fs", time.perf_counter() - _obj_create_start)
        
        # https://modal.com/docs/reference/modal.Queue#ephemeral
        _queue_create_start = time.perf_counter()
        logger.warning("[COLD_START] Creating ephemeral Modal queue...")
        with modal.Queue.ephemeral(client=client) as q:
            logger.warning("[COLD_START] Ephemeral queue created in %.3fs", time.perf_counter() - _queue_create_start)
            logger.info(
                "[COLD_START] Spawning webrtc modal function from %s into modal app %s",
                RTCPeerConnectionModal.__name__,
                app.name,
            )
            # https://modal.com/docs/reference/modal.Function#spawn
            _spawn_start = time.perf_counter()
            logger.warning("[COLD_START] Calling .spawn() on modal function...")
            function_call: modal.FunctionCall = (
                rtc_modal_obj.rtc_peer_connection_modal.spawn(
                    webrtc_request=webrtc_request,
                    q=q,
                )
            )
            logger.warning("[COLD_START] Modal function .spawn() returned in %.3fs", time.perf_counter() - _spawn_start)
            
            logger.info("[COLD_START] Total time before waiting for response: %.3fs", time.perf_counter() - _spawn_total_start)
            
            try:
                _queue_get_start = time.perf_counter()
                logger.warning("[COLD_START] Waiting for response from Modal queue (timeout=%ss)...", WEBRTC_MODAL_RESPONSE_TIMEOUT)
                answer = WebRTCWorkerResult.model_validate(
                    q.get(block=True, timeout=WEBRTC_MODAL_RESPONSE_TIMEOUT)
                )
                logger.info("[COLD_START] Response received from queue in %.3fs", time.perf_counter() - _queue_get_start)
            except Empty:
                logger.error("[COLD_START] Modal function call timed out after %.3fs, terminating containers", time.perf_counter() - _queue_get_start)
                function_call.cancel(terminate_containers=True)
                raise RoboflowAPITimeoutError("Modal function call timed out")
            except Exception as exc:
                logger.error("[COLD_START] Exception while waiting for response: %s", exc)
                raise exc
            
            logger.info("[COLD_START] Total spawn_rtc_peer_connection_modal completed in %.3fs", time.perf_counter() - _spawn_total_start)
            logger.info("[COLD_START] ========== SPAWN RTC PEER CONNECTION END ==========")
            return answer
