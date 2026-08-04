import asyncio
import datetime
import os
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Callable, Dict, List, Optional

from inference.core import logger
from inference.core.env import (
    ALLOW_CUSTOM_PYTHON_EXECUTION_IN_WORKFLOWS,
    INTERNAL_WEIGHTS_URL_SUFFIX,
    LEGACY_MMP_ADAPTER_BUNDLED_BACKEND,
    LEGACY_MMP_ADAPTER_ENABLED,
    LEGACY_MMP_ADAPTER_MODE,
    LOG_LEVEL,
    MAX_ACTIVE_MODELS,
    MMP_PERFORMANCE_PROFILING_ENABLED,
    MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S,
    MMP_PERFORMANCE_PROFILING_MAX_SAMPLES,
    MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N,
    MMP_PERFORMANCE_PROFILING_WARMUP_CALLS,
    MODAL_TOKEN_ID,
    MODAL_TOKEN_SECRET,
    MODAL_WEB_ENDPOINT_URL,
    MODAL_WORKSPACE_NAME,
    MODAL_WS_ENDPOINT_URL,
    MODEL_CACHE_DIR,
    MODELS_CACHE_AUTH_CACHE_MAX_SIZE,
    MODELS_CACHE_AUTH_CACHE_TTL,
    MODELS_CACHE_AUTH_ENABLED,
    PROJECT,
    ROBOFLOW_INTERNAL_SERVICE_SECRET,
    WEBEXEC_TRANSPORT,
    WEBEXEC_WS_CONNECTION_POOL_SIZE,
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
    WEBRTC_MODAL_ROUTING_REGION,
    WEBRTC_MODAL_RTSP_PLACEHOLDER,
    WEBRTC_MODAL_RTSP_PLACEHOLDER_URL,
    WEBRTC_MODAL_SHUTDOWN_RESERVE,
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WEBRTC_MODAL_USAGE_QUOTA_ENABLED,
    WEBRTC_MODAL_VOLUME_NAME,
    WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
    WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS,
    WEBRTC_SESSION_HEARTBEAT_URL,
    WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
)
from inference.core.exceptions import (
    RoboflowAPITimeoutError,
    RoboflowAPIUnsuccessfulRequestError,
)
from inference.core.interfaces.camera.source_reference_sanitizer import (
    sanitize_source_reference,
)
from inference.core.interfaces.webrtc_worker.entities import (
    WebRTCWorkerRequest,
    WebRTCWorkerResult,
)
from inference.core.interfaces.webrtc_worker.request_utils import (
    reuse_resolved_workspace_id_for_webrtc_request,
)
from inference.core.interfaces.webrtc_worker.utils import (
    warmup_cuda,
    workflow_contains_instant_model,
    workflow_contains_preloaded_model,
)
from inference.core.interfaces.webrtc_worker.watchdog import Watchdog
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.registries.roboflow import RoboflowModelRegistry
from inference.core.roboflow_api import get_workflow_specification
from inference.core.version import __version__
from inference.models.aliases import resolve_roboflow_model_alias
from inference.models.owlv2.owlv2 import PRELOADED_HF_MODELS, preload_owlv2_model
from inference.models.utils import ROBOFLOW_MODEL_TYPES
from inference.usage_tracking.collector import usage_collector
from inference.usage_tracking.plan_details import WebRTCPlan

try:
    import modal
except ImportError:
    modal = None


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


_mmp_handle = None
_mmp_started = False
_mmp_lock = threading.Lock()


def _ensure_mmp_started() -> None:
    """Start the in-container ModelManagerProcess once per container.

    Started lazily on the first call instead of @modal.enter so the live MMP
    thread, its SHM pool and ZMQ socket never land in a memory snapshot.
    """
    global _mmp_handle, _mmp_started
    with _mmp_lock:
        if _mmp_started:
            # A cancelled first attempt may have left the MMP thread alive with
            # no handle recorded; relaunching would collide on the same address,
            # so the one launch attempt per container stands either way.
            return
        _mmp_started = True
        os.environ.setdefault("INFERENCE_MMP_ADDR", "ipc:///tmp/webrtc_mmp")
        from inference_server.launcher import launch_orchestrated

        _mmp_handle = launch_orchestrated(mmp_addr=os.environ["INFERENCE_MMP_ADDR"])
        logger.info(
            "MMP started for adapter: addr=%s shm=%s",
            _mmp_handle.mmp_addr,
            _mmp_handle.shm_name,
        )


_adapter = None
_adapter_lock = threading.Lock()


def _ensure_adapter_started():
    """Container-singleton ModelManagerAdapter on a persistent background loop.

    One adapter per container: per-call construction would rebuild the whole
    in-process ModelManager in bundled mode, and asyncio locks/futures cannot
    be shared across per-call asyncio.run loops. The adapter's sync bridge is
    callable from any other thread. In bundled+direct mode the models loaded
    through this adapter live in THIS process, so a snap=True preload lands
    them inside the memory snapshot.
    """
    global _adapter
    if LEGACY_MMP_ADAPTER_MODE != "bundled":
        _ensure_mmp_started()
    with _adapter_lock:
        if _adapter is not None:
            return _adapter
        from inference.core.managers.mmp_adapter import ModelManagerAdapter

        legacy_stack = WithFixedSizeCache(
            ModelManager(model_registry=RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)),
            max_size=MAX_ACTIVE_MODELS,
        )
        adapter = ModelManagerAdapter(legacy_stack=legacy_stack)
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever, name="mmp-adapter-loop", daemon=True
        )
        thread.start()
        start_state: dict = {}

        async def _start() -> None:
            start_state["task"] = asyncio.current_task()
            await adapter.start()

        start_future = asyncio.run_coroutine_threadsafe(_start(), loop)
        try:
            start_future.result(timeout=30)
        except BaseException:
            # A failed start must not leak the loop thread or a half-started
            # client — the next attempt builds fresh. One sequential cleanup
            # coroutine cancels AND awaits the start task before shutting the
            # adapter down, so nothing interleaves with start's own cleanup.
            async def _cleanup() -> None:
                task = start_state.get("task")
                if task is not None and not task.done():
                    task.cancel()
                    try:
                        await task
                    except BaseException:
                        pass
                await adapter.shutdown()

            try:
                asyncio.run_coroutine_threadsafe(_cleanup(), loop).result(timeout=15)
            except Exception:
                logger.warning("Adapter cleanup after failed start", exc_info=True)
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=5)
            if thread.is_alive():
                logger.warning(
                    "Adapter loop thread did not stop; leaving daemon thread"
                )
            else:
                loop.close()
            raise
        _adapter = adapter
        return _adapter


def _preload_models_over_adapter(adapter, model_ids: List[str]) -> None:
    for model_id in model_ids:
        try:
            de_aliased = resolve_roboflow_model_alias(model_id=model_id)
            adapter.add_model(
                de_aliased,
                api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY or "",
                countinference=False,
                service_secret=ROBOFLOW_INTERNAL_SERVICE_SECRET,
            )
            logger.info("Preloaded %s over adapter", de_aliased)
        except Exception as exc:
            logger.error("Failed to preload %s over adapter: %s", model_id, exc)


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

    video_processing_image = (
        video_processing_image.apt_install("ffmpeg")
        .pip_install("modal", "msgpack", "websocket-client")
        .entrypoint([])
    )

    # https://modal.com/docs/reference/modal.Volume
    rfcache_volume = modal.Volume.from_name(
        WEBRTC_MODAL_VOLUME_NAME, create_if_missing=True
    )

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
            "LEGACY_MMP_ADAPTER_ENABLED": str(LEGACY_MMP_ADAPTER_ENABLED),
            "LEGACY_MMP_ADAPTER_MODE": LEGACY_MMP_ADAPTER_MODE,
            "LEGACY_MMP_ADAPTER_BUNDLED_BACKEND": LEGACY_MMP_ADAPTER_BUNDLED_BACKEND,
            # The snap=True preload gate reads this in-container; without the
            # forward it would default to enabled and skip preload on
            # snapshots-disabled deployments.
            "WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT": str(
                WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT
            ),
            "METRICS_ENABLED": "False",
            "MODAL_TOKEN_ID": MODAL_TOKEN_ID,
            "MODAL_TOKEN_SECRET": MODAL_TOKEN_SECRET,
            "MODAL_WEB_ENDPOINT_URL": MODAL_WEB_ENDPOINT_URL,
            "MODAL_WORKSPACE_NAME": MODAL_WORKSPACE_NAME,
            "MODAL_WS_ENDPOINT_URL": MODAL_WS_ENDPOINT_URL,
            "MODEL_CACHE_DIR": MODEL_CACHE_DIR,
            "MODELS_CACHE_AUTH_CACHE_MAX_SIZE": str(MODELS_CACHE_AUTH_CACHE_MAX_SIZE),
            "MODELS_CACHE_AUTH_CACHE_TTL": str(MODELS_CACHE_AUTH_CACHE_TTL),
            "MODELS_CACHE_AUTH_ENABLED": str(MODELS_CACHE_AUTH_ENABLED),
            "LOG_LEVEL": LOG_LEVEL,
            "MMP_PERFORMANCE_PROFILING_ENABLED": str(MMP_PERFORMANCE_PROFILING_ENABLED),
            "MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S": str(
                MMP_PERFORMANCE_PROFILING_LOG_INTERVAL_S
            ),
            "MMP_PERFORMANCE_PROFILING_MAX_SAMPLES": str(
                MMP_PERFORMANCE_PROFILING_MAX_SAMPLES
            ),
            "MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N": str(
                MMP_PERFORMANCE_PROFILING_SAMPLE_EVERY_N
            ),
            "MMP_PERFORMANCE_PROFILING_WARMUP_CALLS": str(
                MMP_PERFORMANCE_PROFILING_WARMUP_CALLS
            ),
            "ONNXRUNTIME_EXECUTION_PROVIDERS": "[CUDAExecutionProvider,CPUExecutionProvider]",
            "PROJECT": PROJECT,
            "PYTHONASYNCIODEBUG": str(os.getenv("PYTHONASYNCIODEBUG", "0")),
            "ROBOFLOW_ENVIRONMENT": (
                "prod" if PROJECT == "roboflow-platform" else "staging"
            ),
            "ROBOFLOW_INTERNAL_SERVICE_NAME": WEBRTC_MODAL_ROBOFLOW_INTERNAL_SERVICE_NAME,
            "ROBOFLOW_INTERNAL_SERVICE_SECRET": ROBOFLOW_INTERNAL_SERVICE_SECRET,
            "WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE": WORKFLOWS_CUSTOM_PYTHON_EXECUTION_MODE,
            "WEBEXEC_TRANSPORT": WEBEXEC_TRANSPORT,
            "WEBEXEC_WS_CONNECTION_POOL_SIZE": str(WEBEXEC_WS_CONNECTION_POOL_SIZE),
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
            "WEBRTC_SESSION_HEARTBEAT_URL": (
                WEBRTC_SESSION_HEARTBEAT_URL if WEBRTC_SESSION_HEARTBEAT_URL else ""
            ),
            "WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS": str(
                WEBRTC_SESSION_HEARTBEAT_INTERVAL_SECONDS
            ),
        },
        "volumes": {MODEL_CACHE_DIR: rfcache_volume},
    }

    # with_options() cannot set routing_region, so it must be baked into the
    # class decorator at deploy time
    if WEBRTC_MODAL_ROUTING_REGION:
        decorator_kwargs["routing_region"] = WEBRTC_MODAL_ROUTING_REGION

    async def run_rtc_peer_connection_with_watchdog(
        webrtc_request: WebRTCWorkerRequest,
        send_answer: Callable[[WebRTCWorkerResult], None],
        model_manager: ModelManager,
        watchdog: Watchdog,
    ):
        from inference.core.interfaces.webrtc_worker.webrtc import (
            init_rtc_peer_connection_with_loop,
        )

        if LEGACY_MMP_ADAPTER_ENABLED:
            try:
                # Container-singleton adapter on its own persistent loop —
                # instant after the first call (or after snap=True preload).
                model_manager = _ensure_adapter_started()
            except Exception as exc:
                error_msg = f"MMP adapter startup failed: {exc!r}"
                logger.error(error_msg)
                send_answer(WebRTCWorkerResult(error_message=error_msg))
                return

        rtc_peer_connection_task = None
        try:
            rtc_peer_connection_task = asyncio.create_task(
                init_rtc_peer_connection_with_loop(
                    webrtc_request=webrtc_request,
                    send_answer=send_answer,
                    model_manager=model_manager,
                    heartbeat_callback=watchdog.heartbeat,
                    connection_established_callback=watchdog.mark_connection_established,
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

            await rtc_peer_connection_task
            logger.info("Task completed uninterrupted")
        except modal.exception.InputCancellation:
            logger.warning("Modal function was cancelled")
        except asyncio.CancelledError as exc:
            logger.warning("WebRTC connection task was cancelled (%s)", exc)
        except Exception as exc:
            logger.error(exc)
            # Setup failures (e.g. watchdog.start) never reach the peer
            # connection's own answer path; an extra queue item is harmless.
            send_answer(
                WebRTCWorkerResult(error_message=f"WebRTC worker failed: {exc}")
            )
        finally:
            watchdog.stop()
            if (
                rtc_peer_connection_task is not None
                and not rtc_peer_connection_task.done()
            ):
                # Setup failures after task creation land here with the task
                # still running — drain it before the loop closes.
                rtc_peer_connection_task.cancel()
                try:
                    await rtc_peer_connection_task
                except BaseException:
                    pass

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
            _workspace_id = reuse_resolved_workspace_id_for_webrtc_request(
                webrtc_request
            )

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
            logger.info(
                "rtsp_url: %s",
                sanitize_source_reference(webrtc_request.rtsp_url or ""),
            )
            logger.info("processing_timeout: %s", webrtc_request.processing_timeout)
            logger.info("requested_region: %s", webrtc_request.requested_region)
            logger.info("watchdog_timeout: %s", WEBRTC_MODAL_WATCHDOG_TIMEMOUT)
            logger.info("requested_plan: %s", webrtc_request.requested_plan)
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
            logger.info("MODAL_IDENTITY_TOKEN set: %s", bool(MODAL_IDENTITY_TOKEN))

            performance_profiler = None
            if MMP_PERFORMANCE_PROFILING_ENABLED:
                from inference_models.utils.performance import performance_profiler

                performance_profiler.set_metadata("modal.image_tag", docker_tag)
                performance_profiler.set_metadata("modal.gpu", self._gpu)
                performance_profiler.set_metadata("modal.workflow_id", workflow_id)
                performance_profiler.set_metadata("modal.cold_start", self._cold_start)
                performance_profiler.set_metadata(
                    "modal.function_call_number",
                    self._function_call_number_on_container,
                )
                performance_profiler.set_metadata(
                    "modal.memory_snapshot_enabled",
                    WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT,
                )
                performance_profiler.set_metadata(
                    "modal.preload_models", self.preload_models
                )

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

            watchdog = Watchdog(
                api_key=webrtc_request.api_key,
                timeout_seconds=WEBRTC_MODAL_WATCHDOG_TIMEMOUT,
                workspace_id=getattr(webrtc_request, "workspace_id", None),
                session_id=getattr(webrtc_request, "session_id", None),
                heartbeat_url=WEBRTC_SESSION_HEARTBEAT_URL,
            )

            if LEGACY_MMP_ADAPTER_ENABLED:
                try:
                    _ensure_adapter_started()
                except Exception as exc:
                    error_msg = f"MMP adapter startup failed: {exc}"
                    logger.error(error_msg)
                    send_answer(WebRTCWorkerResult(error_message=error_msg))
                    return

            try:
                asyncio.run(
                    run_rtc_peer_connection_with_watchdog(
                        webrtc_request=webrtc_request,
                        send_answer=send_answer,
                        model_manager=self._model_manager,
                        watchdog=watchdog,
                    )
                )
            except modal.exception.InputCancellation:
                logger.warning("Modal function was cancelled")
            except asyncio.CancelledError as exc:
                logger.warning("WebRTC connection task was cancelled (%s)", exc)
            except Exception as exc:
                logger.warning("Unhandled exception: %s", exc)
            finally:
                watchdog.stop()
                if performance_profiler is not None:
                    try:
                        performance_profiler.flush(force=True)
                    except Exception:
                        logger.warning(
                            "Could not flush Modal performance profile",
                            exc_info=True,
                        )

            _exec_session_stopped = datetime.datetime.now()
            logger.info(
                "WebRTC session stopped at %s",
                _exec_session_stopped.isoformat(),
            )

            no_frames_processed = watchdog.total_heartbeats == 0

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
                resource_id=workflow_id,
                resource_details={
                    "plan": webrtc_plan,
                    "billable": True,
                    "video_source": video_source,
                    "is_preview": webrtc_request.is_preview,
                },
                execution_duration=(
                    (_exec_session_stopped - _exec_session_started).total_seconds()
                    if watchdog.connection_established
                    else 0
                ),
            )

            logger.info("Function completed")

            if no_frames_processed:
                if watchdog.connection_established:
                    usage_collector.push_usage_payloads()
                    raise Exception(
                        "WebRTC connection was established but no frames were processed. "
                        "This typically indicates an invalid RTSP stream URL or corrupted video file."
                    )
                else:
                    raise Exception(
                        "WebRTC connection could not be established. "
                        "No frames were processed."
                    )
            usage_collector.push_usage_payloads()

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
            # TODO: pre-load models on CPU
            logger.info("Starting CPU container")
            self._gpu = "CPU"
            self._cold_start = False

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
            self._cold_start = False
            time_start = time.time()
            warmup_cuda(max_retries=10, retry_delay=0.5)
            self._gpu = check_nvidia_smi_gpu()
            logger.info("Starting GPU container on %s", self._gpu)
            logger.info("Preload hf ids: %s", self.preload_hf_ids)
            logger.info("Preload models: %s", self.preload_models)
            if LEGACY_MMP_ADAPTER_ENABLED:
                # bundled+direct: models loaded here live in THIS process and
                # are captured by the memory/GPU snapshot — warm restores.
                # mmp / bundled+subprocess load into worker subprocesses a
                # snapshot cannot capture, so with snapshots enabled those
                # modes must start lazily on the first call instead.
                snapshot_safe = (
                    LEGACY_MMP_ADAPTER_MODE == "bundled"
                    and LEGACY_MMP_ADAPTER_BUNDLED_BACKEND == "direct"
                ) or not WEBRTC_MODAL_FUNCTION_ENABLE_MEMORY_SNAPSHOT
                if snapshot_safe:
                    adapter = _ensure_adapter_started()
                    if self.preload_models:
                        preload_models = [
                            m.strip() for m in self.preload_models.split(",")
                        ]
                        logger.info(
                            "Preloading models over adapter: %s", preload_models
                        )
                        _preload_models_over_adapter(adapter, preload_models)
                else:
                    logger.info(
                        "Adapter mode %s/%s is not snapshot-safe: adapter "
                        "starts on first call, models load on first use",
                        LEGACY_MMP_ADAPTER_MODE,
                        LEGACY_MMP_ADAPTER_BUNDLED_BACKEND,
                    )
            if not LEGACY_MMP_ADAPTER_ENABLED and self.preload_hf_ids:
                preload_hf_ids = [m.strip() for m in self.preload_hf_ids.split(",")]
                for preload_hf_id in preload_hf_ids:
                    logger.info("Preloading owlv2 base model: %s", preload_hf_id)
                    preload_owlv2_model(preload_hf_id)
            if not LEGACY_MMP_ADAPTER_ENABLED and self.preload_models:
                preload_models = []
                if self.preload_models:
                    preload_models = [m.strip() for m in self.preload_models.split(",")]
                model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
                model_manager = ModelManager(model_registry=model_registry)
                for model_id in preload_models:
                    try:
                        de_aliased_model_id = resolve_roboflow_model_alias(
                            model_id=model_id
                        )
                        logger.info(f"Preloading model: {de_aliased_model_id}")
                        model_manager.add_model(
                            model_id=de_aliased_model_id,
                            api_key=WEBRTC_MODAL_MODELS_PRELOAD_API_KEY,
                            countinference=False,
                            service_secret=ROBOFLOW_INTERNAL_SERVICE_SECRET,
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to preload model %s: %s",
                            model_id,
                            exc,
                        )
                self._model_manager = model_manager
            time_end = time.time()
            self._container_startup_time_seconds = time_end - time_start

    def spawn_rtc_peer_connection_modal(
        webrtc_request: WebRTCWorkerRequest,
    ) -> WebRTCWorkerResult:
        requested_gpu: Optional[str] = None
        requested_ram_mb: Optional[int] = None
        requested_cpu_cores: Optional[int] = None
        webrtc_plans: Optional[Dict[str, WebRTCPlan]] = (
            usage_collector._plan_details.get_webrtc_plans(
                api_key=webrtc_request.api_key
            )
        )
        if webrtc_plans and webrtc_request.requested_plan:
            if webrtc_request.requested_plan not in webrtc_plans:
                raise RoboflowAPIUnsuccessfulRequestError(
                    f"Unknown requested plan {webrtc_request.requested_plan}, available plans: {', '.join(webrtc_plans.keys())}"
                )
            requested_gpu = webrtc_plans[webrtc_request.requested_plan].gpu
            requested_ram_mb = webrtc_plans[webrtc_request.requested_plan].ram_mb
            requested_cpu_cores = webrtc_plans[webrtc_request.requested_plan].cpu_cores

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

        workspace_id = reuse_resolved_workspace_id_for_webrtc_request(webrtc_request)
        if not webrtc_request.workflow_configuration.workflow_specification:
            webrtc_request.workflow_configuration.workflow_specification = get_workflow_specification(
                api_key=webrtc_request.api_key,
                workspace_id=webrtc_request.workflow_configuration.workspace_name,
                workflow_id=webrtc_request.workflow_configuration.workflow_id,
                workflow_version_id=webrtc_request.workflow_configuration.workflow_version_id,
            )
        tags = {"tag": docker_tag}
        if workspace_id:
            tags["workspace_id"] = workspace_id

        # TODO: tag function run

        # Modal parametrization
        preload_hf_ids = ""
        if WEBRTC_MODAL_PRELOAD_HF_IDS and workflow_contains_instant_model(
            workflow_specification=webrtc_request.workflow_configuration.workflow_specification
        ):
            logger.info("Parametrized preload hf ids: %s", WEBRTC_MODAL_PRELOAD_HF_IDS)
            preload_hf_ids = WEBRTC_MODAL_PRELOAD_HF_IDS

        preload_models = ""
        if WEBRTC_MODAL_PRELOAD_MODELS and workflow_contains_preloaded_model(
            workflow_specification=webrtc_request.workflow_configuration.workflow_specification,
            preload_models=[m.strip() for m in WEBRTC_MODAL_PRELOAD_MODELS.split(",")],
        ):
            logger.info("Parametrized preload models: %s", WEBRTC_MODAL_PRELOAD_MODELS)
            preload_models = WEBRTC_MODAL_PRELOAD_MODELS

        if requested_gpu:
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
        if requested_gpu is not None:
            logger.info(
                "Spawning webrtc modal function with gpu %s",
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
                "Spawning webrtc modal function with region %s",
                webrtc_request.requested_region,
            )
            cls_with_options = cls_with_options.with_options(
                region=webrtc_request.requested_region,
            )
        if requested_ram_mb is not None:
            logger.info(
                "Spawning webrtc modal function with ram %s",
                requested_ram_mb,
            )
            cls_with_options = cls_with_options.with_options(
                memory=requested_ram_mb,
            )
        if requested_cpu_cores is not None:
            logger.info(
                "Spawning webrtc modal function with cpu cores %s",
                requested_cpu_cores,
            )
            cls_with_options = cls_with_options.with_options(
                cpu=requested_cpu_cores,
            )
        rtc_modal_obj: RTCPeerConnectionModal = cls_with_options(
            preload_hf_ids=preload_hf_ids,
            preload_models=preload_models,
        )
        # https://modal.com/docs/reference/modal.Queue#ephemeral
        with modal.Queue.ephemeral(client=client) as q:
            logger.info(
                "Spawning webrtc modal function from %s into modal app %s",
                RTCPeerConnectionModal.__name__,
                app.name,
            )
            # https://modal.com/docs/reference/modal.Function#spawn
            function_call: modal.FunctionCall = (
                rtc_modal_obj.rtc_peer_connection_modal.spawn(
                    webrtc_request=webrtc_request,
                    q=q,
                )
            )
            try:
                answer = WebRTCWorkerResult.model_validate(
                    q.get(block=True, timeout=WEBRTC_MODAL_RESPONSE_TIMEOUT)
                )
            except Empty:
                logger.error("Modal function call timed out, cancelling function call")
                try:
                    function_call.cancel()
                except Exception as cancel_exc:
                    logger.warning(
                        "Failed to cancel timed-out Modal function call: %s",
                        cancel_exc,
                    )
                raise RoboflowAPITimeoutError("Modal function call timed out")
            except Exception as exc:
                logger.error(exc)
                raise exc
            return answer
