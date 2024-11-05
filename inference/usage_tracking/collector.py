import asyncio
import atexit
import datetime
import json
import mimetypes
import socket
import sys
import time
from collections import defaultdict
from functools import wraps
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, TypeVar
from uuid import uuid4

from typing_extensions import ParamSpec

from inference.core.env import API_KEY, LAMBDA, REDIS_HOST
from inference.core.logger import logger
from inference.core.version import __version__ as inference_version
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    CompiledWorkflow,
)

from .config import TelemetrySettings, get_telemetry_settings
from .payload_helpers import (
    APIKey,
    APIKeyHash,
    APIKeyUsage,
    ResourceCategory,
    ResourceDetails,
    ResourceID,
    SystemDetails,
    UsagePayload,
    send_usage_payload,
    sha256_hash,
    zip_usage_payloads,
)
from .plan_details import PlanDetails
from .redis_queue import RedisQueue
from .sqlite_queue import SQLiteQueue
from .utils import collect_func_params

T = TypeVar("T")
P = ParamSpec("P")


class UsageCollector:
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        with UsageCollector._lock:
            if not hasattr(cls, "_instance"):
                cls._instance = super().__new__(cls)
                cls._instance._queue = None
        return cls._instance

    def __init__(self):
        with UsageCollector._lock:
            if self._queue:
                return

        # Async lock only for async protection, should not be shared between threads
        self._async_lock = None
        try:
            self._async_lock = asyncio.Lock()
        except Exception as exc:
            logger.debug("Failed to create async lock %s", exc)

        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._usage: APIKeyUsage = self.empty_usage_dict(
            exec_session_id=self._exec_session_id
        )

        self._hashed_api_keys: Dict[APIKey, APIKeyHash] = {}
        self._api_keys_hashing_enabled = True

        self._plan_details = PlanDetails(
            api_plan_endpoint_url=self._settings.api_plan_endpoint_url,
            sqlite_cache_enabled=False,
        )
        if LAMBDA and REDIS_HOST:
            logger.debug("Persistence through RedisQueue")
            self._queue: "Queue[UsagePayload]" = RedisQueue()
            self._api_keys_hashing_enabled = False
        elif LAMBDA or self._settings.opt_out:
            logger.debug("No persistence")
            self._queue: "Queue[UsagePayload]" = Queue(
                maxsize=self._settings.queue_size
            )
            self._api_keys_hashing_enabled = False
        else:
            try:
                self._queue = SQLiteQueue()
                logger.debug("Persistence through SQLiteQueue")
            except Exception as exc:
                logger.debug("Unable to create instance of SQLiteQueue, %s", exc)
                logger.debug("No persistence")
                self._queue: "Queue[UsagePayload]" = Queue(
                    maxsize=self._settings.queue_size
                )
                self._api_keys_hashing_enabled = False
            try:
                self._plan_details = PlanDetails(
                    api_plan_endpoint_url=self._settings.api_plan_endpoint_url,
                )
                logger.debug("Cached plan details")
            except Exception as exc:
                logger.debug("Unable to create instance of SQLiteQueue, %s", exc)
        self._queue_lock = Lock()

        self._system_info_lock = Lock()
        self._system_info: Dict[str, Any] = {}
        self._resource_details_lock = Lock()
        self._resource_details: DefaultDict[
            APIKey, Dict[Tuple[ResourceCategory, ResourceID], Dict[str, Any]]
        ] = defaultdict(dict)

        self._terminate_collector_thread = Event()
        self._collector_thread = Thread(target=self._usage_collector, daemon=True)
        self._collector_thread.start()

        self._terminate_sender_thread = Event()
        self._sender_thread = Thread(target=self._usage_sender, daemon=True)
        self._sender_thread.start()

        atexit.register(self._cleanup)

    @staticmethod
    def empty_usage_dict(exec_session_id: str) -> APIKeyUsage:
        return defaultdict(  # api_key_hash
            lambda: defaultdict(  # category:resource_id
                lambda: {
                    "timestamp_start": None,
                    "timestamp_stop": None,
                    "exec_session_id": exec_session_id,
                    "ip_address_hash": "",
                    "processed_frames": 0,
                    "fps": 0,
                    "source_duration": 0,
                    "category": "",
                    "resource_id": "",
                    "resource_details": "{}",
                    "hosted": LAMBDA,
                    "api_key_hash": "",
                    "is_gpu_available": False,
                    "python_version": sys.version.split()[0],
                    "inference_version": inference_version,
                    "enterprise": False,
                }
            )
        )

    def _dump_usage_queue_no_lock(self) -> List[APIKeyUsage]:
        usage_payloads: List[APIKeyUsage] = []
        while self._queue:
            if self._queue.empty():
                break
            payload = self._queue.get_nowait()
            if not payload:
                continue
            if not isinstance(payload, list):
                payload = [payload]
            usage_payloads.extend(payload)
        return usage_payloads

    def _dump_usage_queue_with_lock(self) -> List[APIKeyUsage]:
        with self._queue_lock:
            usage_payloads = self._dump_usage_queue_no_lock()
        return usage_payloads

    def _calculate_api_key_hash(self, api_key: APIKey) -> APIKeyHash:
        api_key_hash = ""
        if not api_key:
            api_key = API_KEY
        if api_key:
            api_key_hash = self._hashed_api_keys.get(api_key)
            if not api_key_hash:
                if self._api_keys_hashing_enabled:
                    api_key_hash = sha256_hash(api_key)
                else:
                    api_key_hash = api_key
            self._hashed_api_keys[api_key] = api_key_hash
        return api_key_hash

    @staticmethod
    def _calculate_resource_hash(resource_details: Dict[str, Any]) -> str:
        return sha256_hash(json.dumps(resource_details, sort_keys=True))

    def _enqueue_payload(self, payload: UsagePayload):
        logger.debug("Enqueuing usage payload %s", payload)
        if not payload:
            return
        with self._queue_lock:
            if not self._queue.full():
                self._queue.put(payload)
            else:
                usage_payloads = self._dump_usage_queue_no_lock()
                usage_payloads.append(payload)
                merged_usage_payloads = zip_usage_payloads(
                    usage_payloads=usage_payloads,
                )
                for usage_payload in merged_usage_payloads:
                    self._queue.put(usage_payload)

    def record_resource_details(
        self,
        category: str,
        resource_details: Dict[str, Any],
        resource_id: str = "",
        api_key: str = "",
    ):
        if not category:
            raise ValueError("Category is compulsory when recording resource details.")
        if not resource_details or not isinstance(resource_details, dict):
            logger.debug(
                "Tried to record non-dict resource details, '%s'", resource_details
            )
            return

        if not resource_id:
            resource_id = UsageCollector._calculate_resource_hash(
                resource_details=resource_details
            )

        with self._resource_details_lock:
            api_key_resource_details = self._resource_details[api_key]
            api_key_resource_details[(category, resource_id)] = resource_details

    @staticmethod
    def system_info(
        ip_address: Optional[str] = None,
    ) -> SystemDetails:
        if ip_address:
            ip_address_hash_hex = sha256_hash(ip_address)
        else:
            try:
                ip_address: str = socket.gethostbyname(socket.gethostname())
            except:
                s = None
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    ip_address = s.getsockname()[0]
                except:
                    ip_address: str = socket.gethostbyname("localhost")

                if s:
                    s.close()

            ip_address_hash_hex = sha256_hash(ip_address)

        return {
            "ip_address_hash": ip_address_hash_hex,
            "is_gpu_available": False,  # TODO
        }

    def record_system_info(
        self,
        ip_address: Optional[str] = None,
    ):
        if self._system_info:
            return
        with self._system_info_lock:
            self._system_info = self.system_info(
                ip_address=ip_address,
            )
        logger.debug("Usage (system info): %s", self._system_info)

    @staticmethod
    def _guess_source_type(source: str) -> str:
        mime_type, _ = mimetypes.guess_type(source)
        stream_schemes = ["rtsp", "rtmp"]
        source_type = None
        if mime_type and mime_type.startswith("video"):
            source_type = "video"
        elif mime_type and mime_type.startswith("image"):
            source_type = "image"
        elif mime_type:
            logger.debug("Unhandled mime type")
            source_type = mime_type.split("/")[0]
        elif not mime_type and str.isnumeric(source):
            source_type = "camera"
        elif not mime_type and any(
            source.lower().startswith(s) for s in stream_schemes
        ):
            source_type = "stream"
        return source_type

    def _update_usage_payload(
        self,
        source: str,
        category: str,
        frames: int = 1,
        api_key: APIKey = "",
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: str = "",
        inference_test_run: bool = False,
        fps: float = 0,
    ):
        source = str(source) if source else ""
        api_key_hash = self._calculate_api_key_hash(api_key=api_key)
        if not resource_id and resource_details:
            resource_id = UsageCollector._calculate_resource_hash(resource_details)
        with self._resource_details_lock:
            resource_details = self._resource_details.get(api_key, {}).get(
                (category, resource_id), {}
            )
        with self._system_info_lock:
            ip_address_hash = self._system_info["ip_address_hash"]
            is_gpu_available = self._system_info["is_gpu_available"]
        with UsageCollector._lock:
            source_usage = self._usage[api_key_hash][f"{category}:{resource_id}"]
            if not source_usage["timestamp_start"]:
                source_usage["timestamp_start"] = time.time_ns()
            source_usage["timestamp_stop"] = time.time_ns()
            source_usage["processed_frames"] += frames if not inference_test_run else 0
            source_usage["fps"] = round(fps, 2)
            source_usage["source_duration"] += (
                frames / fps if fps and not inference_test_run else 0
            )
            source_usage["category"] = category
            source_usage["resource_id"] = resource_id
            source_usage["resource_details"] = json.dumps(resource_details)
            source_usage["api_key_hash"] = api_key_hash
            source_usage["ip_address_hash"] = ip_address_hash
            source_usage["is_gpu_available"] = is_gpu_available
            logger.debug("Updated usage: %s", source_usage)

    def record_usage(
        self,
        source: str,
        category: str,
        frames: int = 1,
        api_key: APIKey = "",
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: str = "",
        inference_test_run: bool = False,
        fps: float = 0,
    ) -> DefaultDict[str, Any]:
        if not api_key:
            return
        if self._settings.opt_out and not api_key:
            return
        self.record_system_info()
        self.record_resource_details(
            category=category,
            resource_details=resource_details,
            resource_id=resource_id,
            api_key=api_key,
        )
        self._update_usage_payload(
            source=source,
            category=category,
            frames=frames,
            api_key=api_key,
            resource_details=resource_details,
            resource_id=resource_id,
            inference_test_run=inference_test_run,
            fps=fps,
        )

    async def async_record_usage(
        self,
        source: str,
        category: str,
        frames: int = 1,
        api_key: APIKey = "",
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: str = "",
        inference_test_run: bool = False,
        fps: float = 0,
    ) -> DefaultDict[str, Any]:
        if self._async_lock:
            async with self._async_lock:
                self.record_usage(
                    source=source,
                    category=category,
                    frames=frames,
                    api_key=api_key,
                    resource_details=resource_details,
                    resource_id=resource_id,
                    inference_test_run=inference_test_run,
                    fps=fps,
                )
        else:
            self.record_usage(
                source=source,
                category=category,
                frames=frames,
                api_key=api_key,
                resource_details=resource_details,
                resource_id=resource_id,
                inference_test_run=inference_test_run,
                fps=fps,
            )

    def _usage_collector(self):
        while True:
            if self._terminate_collector_thread.wait(self._settings.flush_interval):
                break
            self._enqueue_usage_payload()
        logger.debug("Terminating collector thread")
        self._enqueue_usage_payload()

    def _enqueue_usage_payload(self):
        if not self._usage:
            return
        with UsageCollector._lock:
            self._enqueue_payload(payload=self._usage)
            self._usage = self.empty_usage_dict(exec_session_id=self._exec_session_id)

    def _usage_sender(self):
        while True:
            if self._terminate_sender_thread.wait(self._settings.flush_interval):
                break
            self._flush_queue()
        logger.debug("Terminating sender thread")
        self._flush_queue()

    def _flush_queue(self):
        usage_payloads = self._dump_usage_queue_with_lock()
        if not usage_payloads:
            return
        merged_payloads: APIKeyUsage = zip_usage_payloads(
            usage_payloads=usage_payloads,
        )
        self._offload_to_api(payloads=merged_payloads)

    def _offload_to_api(self, payloads: List[APIKeyUsage]):
        ssl_verify = True
        if "localhost" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False

        hashes_to_api_keys = dict(a[::-1] for a in self._hashed_api_keys.items())

        for payload in payloads:
            for api_key_hash, resource_payloads in payload.items():
                if api_key_hash not in hashes_to_api_keys:
                    logger.debug(
                        "Cannot obtain plan details, api key hash cannot be resolved"
                    )
                    continue
                api_key = hashes_to_api_keys[api_key_hash]
                api_key_plan_details = self._plan_details.get_api_key_plan(
                    api_key=api_key
                )

                for resource_payload in resource_payloads.values():
                    resource_payload["enterprise"] = api_key_plan_details[
                        self._plan_details._is_enterprise_col_name
                    ]

            logger.debug("Sending usage payload %s", payload)
            api_keys_hashes_failed = send_usage_payload(
                payload=payload,
                api_usage_endpoint_url=self._settings.api_usage_endpoint_url,
                hashes_to_api_keys=hashes_to_api_keys,
                ssl_verify=ssl_verify,
            )
            if api_keys_hashes_failed:
                logger.debug(
                    "Failed to send usage following usage payloads: %s",
                    api_keys_hashes_failed,
                )
            for api_key_hash in list(payload.keys()):
                if api_key_hash not in api_keys_hashes_failed:
                    del payload[api_key_hash]
            if payload:
                logger.debug("Enqueuing back unsent payload")
                self._enqueue_payload(payload=payload)

    def push_usage_payloads(self):
        self._enqueue_usage_payload()
        self._flush_queue()

    async def async_push_usage_payloads(self):
        if self._async_lock:
            async with self._async_lock:
                self.push_usage_payloads()
        else:
            self.push_usage_payloads()

    @staticmethod
    def _resource_details_from_workflow_json(
        workflow_json: Dict[str, Any]
    ) -> ResourceDetails:
        return {
            "steps": [
                f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
                for step in workflow_json.get("steps", [])
                if isinstance(step, dict)
            ]
        }

    @staticmethod
    def _extract_usage_params_from_func_kwargs(
        usage_fps: float,
        usage_api_key: str,
        usage_workflow_id: str,
        usage_workflow_preview: bool,
        usage_inference_test_run: bool,
        usage_billable: bool,
        func: Callable[[Any], Any],
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        func_kwargs = collect_func_params(func, args, kwargs)
        resource_details = {
            "billable": usage_billable,
        }
        resource_id = ""
        category = None
        # TODO: add requires_api_key, True if workflow definition comes from platform or model comes from workspace
        if "workflow" in func_kwargs:
            workflow: CompiledWorkflow = func_kwargs["workflow"]
            if hasattr(workflow, "init_parameters"):
                init_parameters = workflow.init_parameters
                if "workflows_core.api_key" in init_parameters:
                    usage_api_key = init_parameters["workflows_core.api_key"]
            workflow_json = {}
            if hasattr(workflow, "workflow_json"):
                if isinstance(workflow.workflow_json, dict):
                    workflow_json = workflow.workflow_json
                else:
                    logger.debug(
                        "Got non-dict workflow JSON, '%s'", workflow.workflow_json
                    )
            new_resource_details = UsageCollector._resource_details_from_workflow_json(
                workflow_json=workflow_json,
            )
            resource_details.update(new_resource_details)
            resource_details["is_preview"] = usage_workflow_preview
            resource_id = usage_workflow_id
            if not resource_id and resource_details:
                usage_workflow_id = UsageCollector._calculate_resource_hash(
                    resource_details=resource_details
                )
            category = "workflows"
        elif "self" in func_kwargs:
            _self = func_kwargs["self"]
            if hasattr(_self, "dataset_id") and hasattr(_self, "version_id"):
                model_id = f"{_self.dataset_id}/{_self.version_id}"
                category = "model"
                resource_id = model_id
            elif isinstance(kwargs, dict) and "model_id" in kwargs:
                model_id = kwargs["model_id"]
                category = "model"
                resource_id = model_id
            else:
                resource_id = "unknown"
                category = "unknown"
            if isinstance(kwargs, dict) and "source" in kwargs:
                resource_details["source"] = kwargs["source"]
            if hasattr(_self, "task_type"):
                resource_details["task_type"] = _self.task_type
        else:
            resource_id = "unknown"
            category = "unknown"

        source = None
        runtime_parameters = func_kwargs.get("runtime_parameters")
        if (
            isinstance(runtime_parameters, dict)
            and "image" in func_kwargs["runtime_parameters"]
        ):
            images = runtime_parameters["image"]
            if not isinstance(images, list):
                images = [images]
            image = images[0]
            if isinstance(image, dict):
                source = image.get("value")
            elif hasattr(image, "_image_reference"):
                source = image._image_reference

        if not usage_api_key:
            _self = func_kwargs.get("self")
            if "api_key" in func_kwargs and func_kwargs["api_key"]:
                usage_api_key = func_kwargs["api_key"]
            elif _self and hasattr(_self, "api_key") and _self.api_key:
                usage_api_key = _self.api_key
            elif (
                "kwargs" in func_kwargs
                and isinstance(func_kwargs["kwargs"], dict)
                and "api_key" in func_kwargs["kwargs"]
                and func_kwargs["kwargs"]["api_key"]
            ):
                usage_api_key = func_kwargs["kwargs"]["api_key"]
            else:
                logger.debug("Could not obtain API key from func kwargs")

        return {
            "source": source,
            "api_key": usage_api_key,
            "category": category,
            "resource_details": resource_details,
            "resource_id": resource_id,
            "inference_test_run": usage_inference_test_run,
            "fps": usage_fps,
        }

    def __call__(self, func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def sync_wrapper(
            *args: P.args,
            usage_fps: float = 0,
            usage_api_key: APIKey = "",
            usage_workflow_id: str = "",
            usage_workflow_preview: bool = False,
            usage_inference_test_run: bool = False,
            usage_billable: bool = True,
            **kwargs: P.kwargs,
        ) -> T:
            self.record_usage(
                **self._extract_usage_params_from_func_kwargs(
                    usage_fps=usage_fps,
                    usage_api_key=usage_api_key,
                    usage_workflow_id=usage_workflow_id,
                    usage_workflow_preview=usage_workflow_preview,
                    usage_inference_test_run=usage_inference_test_run,
                    usage_billable=usage_billable,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )
            )
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(
            *args: P.args,
            usage_fps: float = 0,
            usage_api_key: APIKey = "",
            usage_workflow_id: str = "",
            usage_workflow_preview: bool = False,
            usage_inference_test_run: bool = False,
            usage_billable: bool = True,
            **kwargs: P.kwargs,
        ) -> T:
            await self.async_record_usage(
                **self._extract_usage_params_from_func_kwargs(
                    usage_fps=usage_fps,
                    usage_api_key=usage_api_key,
                    usage_workflow_id=usage_workflow_id,
                    usage_workflow_preview=usage_workflow_preview,
                    usage_inference_test_run=usage_inference_test_run,
                    usage_billable=usage_billable,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )
            )
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    def _cleanup(self):
        self._terminate_collector_thread.set()
        self._collector_thread.join()
        self._terminate_sender_thread.set()
        self._sender_thread.join()


usage_collector = UsageCollector()
