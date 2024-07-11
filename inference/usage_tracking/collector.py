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
from typing import Any, Callable, DefaultDict, Dict, List, Optional, Union
from uuid import uuid4

import importlib_metadata
import requests
import torch

from inference.core.env import API_KEY, LAMBDA
from inference.core.logger import logger
from inference.usage_tracking.utils import collect_func_params

from .config import TelemetrySettings, get_telemetry_settings

WorkflowID = str
WorkflowUsage = DefaultDict[WorkflowID, Any]
APIKey = str
APIKeyUsage = DefaultDict[APIKey, WorkflowUsage]
WorkflowDetails = Dict[str, Any]
EnvironmentDetails = Dict[str, Any]
UsagePayload = Union[APIKeyUsage, WorkflowDetails, EnvironmentDetails]


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

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._usage: APIKeyUsage = self.empty_usage_dict()

        # TODO: use persistent queue, i.e. https://pypi.org/project/persist-queue/
        self._queue: "Queue[UsagePayload]" = Queue(maxsize=self._settings.queue_size)

        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"

        self._system_info_sent: bool = False
        self._workflow_specifications: DefaultDict[
            APIKey, Dict[WorkflowID[Dict[str, Any]]]
        ] = defaultdict(dict)

        self._terminate_threads = Event()

        self._collector_thread = Thread(target=self._usage_collector, daemon=True)
        self._collector_thread.start()
        self._sender_thread = Thread(target=self._usage_sender, daemon=True)
        self._sender_thread.start()

        atexit.register(self._cleanup)

    @staticmethod
    def empty_usage_dict(exec_session_id: str) -> APIKeyUsage:
        return defaultdict(  # API key
            lambda: defaultdict(  # workflow ID
                lambda: {
                    "timestamp_start": None,
                    "timestamp_stop": None,
                    "exec_session_id": exec_session_id,
                    "processed_frames": 0,
                    "fps": 0,
                    "source_duration": 0,
                    "workflow_id": "",
                    "api_key": None,
                }
            )
        )

    @staticmethod
    def _merge_usage_dicts(d1: UsagePayload, d2: UsagePayload):
        merged = {}
        if (
            "workflow_id" in d1
            and "workflow_id" in d2
            and d1["workflow_id"] != d2["workflow_id"]
        ):
            raise ValueError("Cannot merge usage for different workflow IDs")
        if "api_key" in d1 and "api_key" in d2 and d1["api_key"] != d2["api_key"]:
            raise ValueError("Cannot merge usage for different API keys")
        if "timestamp_start" in d1 and "timestamp_start" in d2:
            merged["timestamp_start"] = min(
                d1["timestampt_start"], d2["timestampt_start"]
            )
        if "timestamp_stop" in d1 and "timestamp_stop" in d2:
            merged["timestamp_stop"] = min(d1["timestamp_stop"], d2["timestamp_stop"])
        if "processed_frames" in d1 and "processed_frames" in d2:
            merged["processed_frames"] = d1["processed_frames"] + d2["processed_frames"]
        return {**d1, **d2, **merged}

    def _dump_and_zip_usage_queue(self) -> APIKeyUsage:
        with UsageCollector._lock:
            usage_payloads: List[APIKeyUsage] = []
            while self._queue:
                if self._queue.empty():
                    break
                usage_payloads.append(self._queue.get_nowait())

        merged_payloads: APIKeyUsage = {}
        environment_details_payload = {}
        for api_workflow_payloads in usage_payloads:
            if "timestamp" in api_workflow_payloads:
                if "workflow_id" not in api_workflow_payloads:
                    environment_details_payload = api_workflow_payloads
                    continue
                api_key = api_workflow_payloads["api_key"]
                workflow_id = api_workflow_payloads["workflow_id"]
                if api_key not in merged_payloads:
                    merged_payloads[api_key] = {}
                if workflow_id not in merged_payloads[api_key]:
                    merged_payloads[api_key][workflow_id] = {}
                merged_payloads[api_key][workflow_id] = (
                    UsageCollector._merge_usage_dicts(
                        merged_payloads[api_key][workflow_id],
                        api_workflow_payloads,
                    )
                )
                continue
            for api_key, workflow_payloads in api_workflow_payloads.items():
                if api_key not in merged_payloads:
                    merged_payloads[api_key] = {}
                merged_api_key_usage = merged_payloads[api_key]
                for workflow_id, usage_payload in workflow_payloads.items():
                    if workflow_id not in merged_api_key_usage:
                        merged_api_key_usage[workflow_id] = {}
                    merged_api_key_usage[workflow_id] = (
                        UsageCollector._merge_usage_dicts(
                            merged_api_key_usage[workflow_id],
                            usage_payload,
                        )
                    )
        if environment_details_payload:
            if environment_details_payload["api_key"] not in merged_payloads:
                merged_payloads[environment_details_payload["api_key"]] = {}
            if not merged_payloads[environment_details_payload["api_key"]]:
                merged_payloads[environment_details_payload["api_key"]][None] = {}
                workflow_usage_payload_key = None
            else:
                workflow_usage_payload_key = next(
                    iter(merged_payloads[environment_details_payload["api_key"]].keys())
                )
            merged_payloads[environment_details_payload["api_key"]][
                workflow_usage_payload_key
            ] = UsageCollector._merge_usage_dicts(
                merged_payloads[environment_details_payload["api_key"]][
                    workflow_usage_payload_key
                ],
                environment_details_payload,
            )
        return merged_payloads

    @staticmethod
    def _hash(payload: str, length=5):
        payload_hash = hashlib.sha256(payload.encode())
        return payload_hash.hexdigest()[:length]

    def _enqueue_payload(self, payload: UsagePayload):
        with UsageCollector._lock:
            if not self._queue.full():
                self._queue.put(payload)
            else:
                # TODO: aggregate
                self._queue.get()
                self._queue.put(payload)

    @staticmethod
    def _calculate_workflow_hash(workflow: Dict[str, Any]) -> str:
        return UsageCollector._hash(json.dumps(workflow, sort_keys=True))

    def record_workflow_specification(
        self,
        workflow_specification: Dict[str, Any],
        workflow_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if not workflow_specification and not workflow_id:
            return
        if not isinstance(workflow_specification, dict) and not workflow_id:
            return

        if not api_key:
            api_key = API_KEY
        if not workflow_id:
            workflow_id = UsageCollector._calculate_workflow_hash(
                workflow=workflow_specification
            )

        with UsageCollector._lock:
            api_key_specifications = self._workflow_specifications[api_key]
            if workflow_id in api_key_specifications:
                logger.debug("Attempt to add workflow specification multiple times.")
                return
            api_key_specifications[workflow_id] = True

        steps_summary = [
            f"{step.get('type', 'unknown')}:{step.get('name', 'unknown')}"
            for step in workflow_specification.get("steps", [])
            if isinstance(step, dict)
        ]
        workflow_specification_payload: WorkflowDetails = {
            "timestamp_start": time.time_ns(),
            "workflow_id": workflow_id,
            "steps": steps_summary,
            "api_key": api_key,
        }
        logger.debug(
            "Usage (workflow specification): %s", workflow_specification_payload
        )
        self._enqueue_payload(payload=workflow_specification_payload)

    @staticmethod
    def system_info(
        exec_session_id,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> EnvironmentDetails:
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
            "ip_address_hash": ip_address_hash_hex,
            "api_key": api_key,
            "is_gpu_available": torch.cuda.is_available(),
            "python_version": sys.version.split()[0],
            "inference_version": inference_version,
        }

    def record_system_info(
        self,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        if self._system_info_sent:
            return
        if not api_key:
            api_key = API_KEY
        execution_details_payload = self.system_info(
            exec_session_id=self._exec_session_id,
            api_key=api_key,
            ip_address=ip_address,
        )
        logger.debug("Usage (execution details): %s", execution_details_payload)
        self._enqueue_payload(payload=execution_details_payload)
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
        frames: int = 1,
        api_key: Optional[str] = None,
        workflow: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ):
        source = str(source) if source else ""
        if not api_key:
            api_key = API_KEY
        if not workflow:
            workflow = self._workflow_specifications[api_key].get(workflow_id)
        if not workflow_id:
            workflow_id = (
                UsageCollector._calculate_workflow_hash(workflow) if workflow else None
            )
        with UsageCollector._lock:
            source_usage = self._usage[api_key][workflow_id]
            if not source_usage["timestamp_start"]:
                source_usage["timestamp_start"] = time.time_ns()
            source_usage["timestamp_stop"] = time.time_ns()
            source_usage["processed_frames"] += frames
            source_usage["fps"] = round(fps, 2)
            source_usage["source_duration"] += frames / fps if fps else 0
            source_usage["workflow_id"] = workflow_id
            source_usage["api_key"] = api_key

    def record_usage(
        self,
        source: str,
        frames: int = 1,
        api_key: Optional[str] = None,
        workflow: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ) -> DefaultDict[str, Any]:
        self.record_system_info(
            api_key=api_key,
        )
        self.record_workflow_specification(
            workflow_specification=workflow,
            workflow_id=workflow_id,
            api_key=api_key,
        )
        self._update_usage_payload(
            source=source,
            frames=frames,
            api_key=api_key,
            workflow=workflow,
            workflow_id=workflow_id,
            fps=fps,
        )

    async def async_record_usage(
        self,
        source: str,
        frames: int = 1,
        api_key: Optional[str] = None,
        workflow: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
        fps: Optional[float] = 0,
    ) -> DefaultDict[str, Any]:
        async with UsageCollector._async_lock:
            self.record_usage(
                source=source,
                frames=frames,
                api_key=api_key,
                workflow=workflow,
                workflow_id=workflow_id,
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
            self._usage = self.empty_usage_dict()

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

        api_keys_sent = []
        for api_key, workflow_payloads in payloads.items():
            try:
                response = requests.post(
                    self._settings.api_usage_endpoint_url,
                    json=list(workflow_payloads.values()),
                    verify=ssl_verify,
                    headers={"Authorization": f"Bearer {api_key}"},
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
            del workflow_payloads[api_key]
        if workflow_payloads:
            logger.warning("Enqueuing unsent payloads")
            self._enqueue_payload(payload=workflow_payloads)

    def _flush_queue(self):
        merged_payloads: APIKeyUsage = self._dump_and_zip_usage_queue(
            usage_queue=self._queue
        )
        # TODO: pub/sub
        if not LAMBDA:
            self._offload_to_api(payloads=merged_payloads)

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
        workflow_json = None
        if "workflow" in func_kwargs:
            if hasattr(func_kwargs["workflow"], "workflow_definition"):
                workflow_definition = func_kwargs["workflow"].workflow_definition
                # TODO: handle enterprise blocks here
            if hasattr(func_kwargs["workflow"], "workflow_json"):
                workflow_json = func_kwargs["workflow"].workflow_json
        if not usage_workflow_id and workflow_json:
            usage_workflow_id = UsageCollector._calculate_workflow_hash(
                workflow=workflow_json
            )
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
            "workflow": workflow_json,
            "workflow_id": usage_workflow_id,
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
