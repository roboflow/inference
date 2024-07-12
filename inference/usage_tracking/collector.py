import asyncio
import atexit
import hashlib
import json
import mimetypes
import socket
import sys
import time
from collections import defaultdict
from functools import wraps
from queue import Queue
from threading import Event, Lock, Thread
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import importlib_metadata
import requests
import torch

from inference.core.env import API_KEY, LAMBDA
from inference.core.logger import logger
from inference.usage_tracking.utils import collect_func_params

from .config import TelemetrySettings, get_telemetry_settings

ResourceID = str
ResourceUsage = DefaultDict[ResourceID, Any]
APIKey = str
APIKeyUsage = DefaultDict[APIKey, ResourceUsage]
ResourceDetails = Dict[str, Any]
SystemDetails = Dict[str, Any]
UsagePayload = Union[APIKeyUsage, ResourceDetails, SystemDetails]


class UsageCollector:
    _lock = Lock()
    _async_lock = asyncio.Lock()

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

        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._usage: APIKeyUsage = self.empty_usage_dict(
            exec_session_id=self._exec_session_id
        )

        # TODO: use persistent queue, i.e. https://pypi.org/project/persist-queue/
        self._queue: "Queue[UsagePayload]" = Queue(maxsize=self._settings.queue_size)
        self._queue_lock = Lock()

        self._system_info_sent: bool = False
        self._resource_details_lock = Lock()
        self._resource_details: DefaultDict[
            APIKey, Dict[ResourceID[Dict[str, Any]]]
        ] = defaultdict(dict)

        self._terminate_threads = Event()

        self._collector_thread = Thread(target=self._usage_collector, daemon=True)
        self._collector_thread.start()
        self._sender_thread = Thread(target=self._usage_sender, daemon=True)
        self._sender_thread.start()

        atexit.register(self._cleanup)

    @staticmethod
    def empty_usage_dict(exec_session_id: str) -> APIKeyUsage:
        return defaultdict(  # api_key
            lambda: defaultdict(  # category:resource_id
                lambda: {
                    "timestamp_start": None,
                    "timestamp_stop": None,
                    "exec_session_id": exec_session_id,
                    "processed_frames": 0,
                    "fps": 0,
                    "source_duration": 0,
                    "category": "",
                    "resource_id": "",
                    "resource_details": {},
                    "api_key": None,
                }
            )
        )

    @staticmethod
    def _merge_usage_dicts(d1: UsagePayload, d2: UsagePayload):
        merged = {}
        if d1 and d2 and d1.get("resource_id") != d2.get("resource_id"):
            raise ValueError("Cannot merge usage for different resource IDs")
        if "api_key" in d1 and "api_key" in d2 and d1["api_key"] != d2["api_key"]:
            raise ValueError("Cannot merge usage for different API keys")
        if "timestamp_start" in d1 and "timestamp_start" in d2:
            merged["timestamp_start"] = min(
                d1["timestamp_start"], d2["timestamp_start"]
            )
        if "timestamp_stop" in d1 and "timestamp_stop" in d2:
            merged["timestamp_stop"] = min(d1["timestamp_stop"], d2["timestamp_stop"])
        if "processed_frames" in d1 and "processed_frames" in d2:
            merged["processed_frames"] = d1["processed_frames"] + d2["processed_frames"]
        if "source_duration" in d1 and "source_duration" in d2:
            merged["source_duration"] = d1["source_duration"] + d2["source_duration"]
        return {**d1, **d2, **merged}

    def _dump_usage_queue(self) -> List[APIKeyUsage]:
        with self._queue_lock:
            usage_payloads: List[APIKeyUsage] = []
            while self._queue:
                if self._queue.empty():
                    break
                usage_payloads.append(self._queue.get_nowait())
        return usage_payloads

    @staticmethod
    def _zip_usage_payloads(usage_payloads: List[APIKeyUsage]) -> APIKeyUsage:
        merged_payloads: APIKeyUsage = {}
        system_info_payload = {}
        for usage_payload in usage_payloads:
            if "inference_version" in usage_payload:
                system_info_payload = usage_payload
                continue
            if "resource_details" in usage_payload:
                resource_details_payload = usage_payload
                api_key = resource_details_payload["api_key"]
                resource_id = resource_details_payload["resource_id"]
                merged_api_key_payload = merged_payloads.setdefault(api_key, {})
                merged_resource_payload = merged_api_key_payload.setdefault(
                    resource_id, {}
                )
                merged_api_key_payload[resource_id] = UsageCollector._merge_usage_dicts(
                    merged_resource_payload,
                    resource_details_payload,
                )
                continue

            for api_key, resource_payloads in usage_payload.items():
                merged_api_key_payload = merged_payloads.setdefault(api_key, {})
                for resource_id, resource_usage_payload in resource_payloads.items():
                    merged_resource_payload = merged_api_key_payload.setdefault(
                        resource_id, {}
                    )
                    merged_api_key_payload[resource_id] = (
                        UsageCollector._merge_usage_dicts(
                            merged_resource_payload,
                            resource_usage_payload,
                        )
                    )

        if system_info_payload:
            api_key = system_info_payload["api_key"]
            merged_api_key_payload = merged_payloads.setdefault(api_key, {})
            resource_id = None
            if merged_api_key_payload:
                resource_id = next(iter(merged_api_key_payload.keys()))
            system_info_payload["resource_id"] = resource_id
            category = system_info_payload["category"]
            merged_resource_payload = merged_api_key_payload.setdefault(
                f"{category}:{resource_id}", {}
            )
            merged_api_key_payload[resource_id] = UsageCollector._merge_usage_dicts(
                merged_resource_payload,
                system_info_payload,
            )
        return merged_payloads

    @staticmethod
    def _hash(payload: str, length=5):
        payload_hash = hashlib.sha256(payload.encode())
        return payload_hash.hexdigest()[:length]

    def _enqueue_payload(self, payload: UsagePayload):
        queue_full = False
        with self._queue_lock:
            if not self._queue.full():
                self._queue.put(payload)
            else:
                queue_full = True
        if queue_full:
            usage_payloads = self._dump_usage_queue()
            usage_payloads.append(payload)
            merged_usage_payloads = self._zip_usage_payloads(
                usage_payloads=usage_payloads,
            )
            with self._queue_lock:
                self._queue.put(merged_usage_payloads)

    @staticmethod
    def _calculate_resource_hash(resource_details: Dict[str, Any]) -> str:
        return UsageCollector._hash(json.dumps(resource_details, sort_keys=True))

    def record_resource_details(
        self,
        category: str,
        resource_details: Dict[str, Any],
        resource_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if not category:
            raise ValueError("Category is compulsory when recording resource details.")
        if not resource_details and not resource_id:
            return
        if not isinstance(resource_details, dict) and not resource_id:
            return

        if not api_key:
            api_key = API_KEY
        if not resource_id:
            resource_id = UsageCollector._calculate_resource_hash(
                resource_details=resource_details
            )

        with self._resource_details_lock:
            api_key_specifications = self._resource_details[api_key]
            if resource_id in api_key_specifications:
                logger.debug("Attempt to add resource details multiple times.")
                return
            api_key_specifications[resource_id] = True

        resource_details_payload: ResourceDetails = {
            "timestamp_start": time.time_ns(),
            "category": category,
            "resource_id": resource_id,
            "resource_details": resource_details,
            "api_key": api_key,
        }
        logger.debug("Usage (%s details): %s", category, resource_details_payload)
        self._enqueue_payload(payload=resource_details_payload)

    @staticmethod
    def system_info(
        exec_session_id: str,
        category: str,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> SystemDetails:
        inference_version = "dev"
        try:
            inference_version = importlib_metadata.version("inference")
        except importlib_metadata.PackageNotFoundError:
            pass

        if ip_address:
            ip_address_hash_hex = UsageCollector._hash(ip_address)
        else:
            ip_address: str = socket.gethostbyname(socket.gethostname())
            ip_address_hash_hex = UsageCollector._hash(ip_address)

        if not api_key:
            api_key = API_KEY

        return {
            "timestamp_start": time.time_ns(),
            "exec_session_id": exec_session_id,
            "category": category,
            "ip_address_hash": ip_address_hash_hex,
            "api_key": api_key,
            "is_gpu_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
            "inference_version": inference_version,
        }

    def record_system_info(
        self,
        api_key: str,
        category: str,
        ip_address: Optional[str] = None,
    ):
        if self._system_info_sent:
            return
        if not api_key:
            api_key = API_KEY
        system_info_payload = self.system_info(
            exec_session_id=self._exec_session_id,
            category=category,
            api_key=api_key,
            ip_address=ip_address,
        )
        logger.debug("Usage (system info): %s", system_info_payload)
        self._enqueue_payload(payload=system_info_payload)
        self._system_info_sent = True

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
        api_key: Optional[str] = None,
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ):
        source = str(source) if source else ""
        if not api_key:
            api_key = API_KEY
        if not resource_id and resource_details:
            resource_id = UsageCollector._calculate_resource_hash(resource_details)
        if not resource_details:
            resource_details = self._resource_details[api_key].get(
                f"{category}:{resource_id}"
            )
        with UsageCollector._lock:
            source_usage = self._usage[api_key][f"{category}:{resource_id}"]
            if not source_usage["timestamp_start"]:
                source_usage["timestamp_start"] = time.time_ns()
            source_usage["timestamp_stop"] = time.time_ns()
            source_usage["processed_frames"] += frames
            source_usage["fps"] = round(fps, 2)
            source_usage["source_duration"] += frames / fps if fps else 0
            source_usage["category"] = category
            source_usage["resource_id"] = resource_id
            source_usage["api_key"] = api_key

    def record_usage(
        self,
        source: str,
        category: str,
        frames: int = 1,
        api_key: Optional[str] = None,
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ) -> DefaultDict[str, Any]:
        self.record_system_info(
            api_key=api_key,
            category=category,
        )
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
            fps=fps,
        )

    async def async_record_usage(
        self,
        source: str,
        category: str,
        frames: int = 1,
        api_key: Optional[str] = None,
        resource_details: Optional[Dict[str, Any]] = None,
        resource_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ) -> DefaultDict[str, Any]:
        async with UsageCollector._async_lock:
            self.record_usage(
                source=source,
                category=category,
                frames=frames,
                api_key=api_key,
                resource_details=resource_details,
                resource_id=resource_id,
                fps=fps,
            )

    def _usage_collector(self):
        while True:
            if self._terminate_threads.wait(self._settings.flush_interval):
                break
            self._enqueue_usage_payload()
        logger.debug("Terminating collector thread")
        self._enqueue_usage_payload()

    def _enqueue_usage_payload(self):
        if not self._usage:
            return
        self._enqueue_payload(payload=self._usage)
        with UsageCollector._lock:
            self._usage = self.empty_usage_dict(exec_session_id=self._exec_session_id)

    def _usage_sender(self):
        while True:
            if self._terminate_threads.wait(self._settings.flush_interval):
                break
            self._flush_queue()
        logger.debug("Terminating sender thread")
        self._flush_queue()

    def _offload_to_api(self, payloads: APIKeyUsage):
        ssl_verify = True
        if "localhost" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False

        # TODO: fast ping before attempting to POST

        api_keys_sent = []
        for api_key, workflow_payloads in payloads.items():
            try:
                response = requests.post(
                    self._settings.api_usage_endpoint_url,
                    json=list(workflow_payloads.values()),
                    verify=ssl_verify,
                    headers={"Authorization": f"Bearer {api_key}"},
                    timeout=1,
                )
                api_keys_sent.append(api_key)
            except Exception as exc:
                logger.warning("Failed to send usage - %s", exc)
                continue
            if response.status_code != 200:
                logger.warning(
                    "Failed to send usage - got %s status code (%s)",
                    response.status_code,
                    response.raw,
                )
                continue
        for api_key in api_keys_sent:
            del payloads[api_key]
        if payloads:
            logger.warning("Enqueuing unsent payloads")
            self._enqueue_payload(payload=payloads)

    def _flush_queue(self):
        usage_payloads = self._dump_usage_queue()
        merged_payloads: APIKeyUsage = self._zip_usage_payloads(
            usage_payloads=usage_payloads,
        )
        self._offload_to_api(payloads=merged_payloads)

    @staticmethod
    def _resource_details_from_workflow_json(
        usage_workflow_id: str, workflow_json: Dict[str, Any]
    ) -> Tuple[ResourceID, ResourceDetails]:
        if not isinstance(workflow_json, dict):
            raise ValueError("workflow_json must be dict")
        resource_details = {}
        resource_details = {
            "steps": [
                f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
                for step in workflow_json.get("steps", [])
                if isinstance(step, dict)
            ]
        }
        if not usage_workflow_id and resource_details:
            usage_workflow_id = UsageCollector._calculate_resource_hash(
                resource_details=resource_details
            )
        resource_id: ResourceID = usage_workflow_id
        return resource_id, resource_details

    @staticmethod
    def _extract_usage_params_from_func_kwargs(
        usage_fps: float,
        usage_api_key: str,
        usage_workflow_id: str,
        func: Callable[[Any], Any],
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not usage_api_key:
            usage_api_key = API_KEY
        func_kwargs = collect_func_params(func, args, kwargs)
        resource_details = {}
        resource_id = None
        category = None
        if "workflow" in func_kwargs:
            if hasattr(func_kwargs["workflow"], "workflow_definition"):
                # TODO: handle enterprise blocks here
                workflow_definition = func_kwargs["workflow"].workflow_definition
            workflow_json = {}
            if hasattr(func_kwargs["workflow"], "workflow_json"):
                workflow_json = func_kwargs["workflow"].workflow_json
            resource_id, resource_details = (
                UsageCollector._resource_details_from_workflow_json(
                    usage_workflow_id=usage_workflow_id,
                    workflow_json=workflow_json,
                )
            )
            category = "workflows"
        elif "model_id" in func_kwargs:
            # TODO: handle model
            pass
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
        return {
            "source": source,
            "api_key": usage_api_key,
            "category": category,
            "resource_details": resource_details,
            "resource_id": resource_id,
            "fps": usage_fps,
        }

    def __call__(self, func: Callable[[Any], Any]):
        @wraps(func)
        def sync_wrapper(
            *args,
            usage_fps: float = 0,
            usage_api_key: str = "",
            usage_workflow_id: str = "",
            **kwargs,
        ):
            self.record_usage(
                **self._extract_usage_params_from_func_kwargs(
                    usage_fps=usage_fps,
                    usage_api_key=usage_api_key,
                    usage_workflow_id=usage_workflow_id,
                    func=func,
                    args=args,
                    kwargs=kwargs,
                )
            )
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(
            *args,
            usage_fps: float = 0,
            usage_api_key: str = "",
            usage_workflow_id: str = "",
            **kwargs,
        ):
            await self.async_record_usage(
                **self._extract_usage_params_from_func_kwargs(
                    usage_fps=usage_fps,
                    usage_api_key=usage_api_key,
                    usage_workflow_id=usage_workflow_id,
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
        self._terminate_threads.set()
        self._collector_thread.join()
        self._sender_thread.join()


usage_collector = UsageCollector()
