import asyncio
import atexit
from collections import defaultdict
from functools import wraps
import hashlib
import json
import mimetypes
from queue import Queue
import socket
import sys
import time
from threading import Event, Lock, Thread
from typing import Any, Callable, DefaultDict, Dict, List, Optional
from uuid import uuid4

import importlib_metadata
import requests
import torch

from inference.core.env import API_KEY
from inference.core.logger import logger
from inference.usage_tracking.utils import collect_func_params

from .config import TelemetrySettings, get_telemetry_settings

Source = str
UsagePayload = DefaultDict[Source, Any]
WorkflowID = str
WorkflowUsage = DefaultDict[WorkflowID, UsagePayload]
APIKey = str
APIKeyUsage = DefaultDict[APIKey, WorkflowUsage]


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

        self._workflow_specifications: DefaultDict[
            APIKey, Dict[WorkflowID[Dict[str, Any]]]
        ] = defaultdict(dict)

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._usage: APIKeyUsage = self._get_empty_usage_dict()

        # TODO: use persistent queue, i.e. https://pypi.org/project/persist-queue/
        self._queue = Queue(maxsize=self._settings.queue_size)

        _ip_address: str = socket.gethostbyname(socket.gethostname())
        self._ip_address_hash_hex = UsageCollector._hash(_ip_address)
        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"
        self._gpu_available: bool = torch.cuda.is_available()
        try:
            self._inference_version: str = importlib_metadata.version("inference")
        except importlib_metadata.PackageNotFoundError:
            self._inference_version: str = "dev"
        self._execution_details_sent: bool = False

        self._terminate_threads = Event()
        self._collector_thread = Thread(target=self._usage_collector, daemon=True)
        self._collector_thread.start()
        self._sender_thread = Thread(target=self._usage_sender, daemon=True)
        self._sender_thread.start()

        atexit.register(self._cleanup)

    def _get_empty_usage_dict(self) -> APIKeyUsage:
        return defaultdict(  # API key
            lambda: defaultdict(  # workflow ID
                lambda: defaultdict(  # source hash
                    lambda: {
                        "timestamp_start": None,
                        "timestamp_stop": None,
                        "exec_session_id": self._exec_session_id,
                        "source_hash": None,
                        "source_type": None,
                        "processed_frames": 0,
                        "fps": 0,
                        "source_duration": 0,
                        "workflow_id": "",
                        "api_key": None,
                    }
                )
            )
        )

    @staticmethod
    def _hash(payload: str, length=5):
        payload_hash = hashlib.sha256(payload.encode())
        return payload_hash.hexdigest()[:length]

    def _enqueue_payload(self, payload: Dict[str, Any]):
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
        if not api_key:
            api_key = API_KEY
        if not workflow_specification and not workflow_id:
            return
        if not workflow_id:
            workflow_id = UsageCollector._calculate_workflow_hash(
                workflow=workflow_specification
            )
        api_key_specifications = self._workflow_specifications[api_key]
        if workflow_id in api_key_specifications:
            logger.debug("Attempt to add workflow specification multiple times.")
            return
        api_key_specifications[workflow_id] = workflow_specification
        workflow_specification_payload = {
            "timestamp": time.time(),
            "workflow_id": workflow_id,
            "workflow": workflow_specification,
            "api_key": api_key,
        }
        logger.debug(
            "Usage (workflow specification): %s", workflow_specification_payload
        )
        self._enqueue_payload(payload=workflow_specification_payload)

    def _get_system_info(
        self, api_key: Optional[str] = None, ip_address: Optional[str] = None
    ):
        if ip_address:
            ip_address_hash_hex = UsageCollector._hash(ip_address)
        else:
            ip_address_hash_hex = self._ip_address_hash_hex

        if not api_key:
            api_key = API_KEY
        return {
            "timestamp": time.time(),
            "exec_session_id": self._exec_session_id,
            "ip_address_hash": ip_address_hash_hex,
            "api_key": api_key,
            "is_gpu_available": self._gpu_available,
            "python_version": sys.version.split()[0],
            "inference_version": self._inference_version,
        }

    def record_execution_details(
        self,
        api_key: Optional[str] = None,
        ip_address: Optional[str] = None,
    ):
        if self._execution_details_sent:
            return
        if not api_key:
            api_key = API_KEY
        execution_details_payload = self._get_system_info(
            api_key=api_key, ip_address=ip_address
        )
        logger.debug("Usage (execution details): %s", execution_details_payload)
        self._enqueue_payload(payload=execution_details_payload)
        self._execution_details_sent = True

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
        source_type = UsageCollector._guess_source_type(source=source)
        source_hash_hex = UsageCollector._hash(source)
        if not api_key:
            api_key = API_KEY
        if not workflow:
            workflow = self._workflow_specifications[api_key].get(workflow_id)
        if not workflow_id:
            workflow_id = (
                UsageCollector._calculate_workflow_hash(workflow) if workflow else None
            )
        source_usage = self._usage[api_key][workflow_id][source_hash_hex]
        if not source_usage["timestamp_start"]:
            source_usage["timestamp_start"] = time.time_ns()
        source_usage["timestamp_stop"] = time.time_ns()
        source_usage["source_hash"] = source_hash_hex
        source_usage["source_type"] = source_type
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
        self.record_execution_details(
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
        self._usage = self._get_empty_usage_dict()

    def _usage_sender(self):
        while True:
            if self._terminate_threads.wait(self._settings.flush_interval):
                break
            self._flush_queue()
        logger.debug("Terminating sender thread")
        self._flush_queue()

    def _flush_queue(self):
        with UsageCollector._lock:
            payloads_by_api_key = {}
            while self._queue:
                if self._queue.empty():
                    break
                item = self._queue.get_nowait()
                if "timestamp" in item:
                    payloads_by_api_key.setdefault(item["api_key"], []).append(item)
                    continue
                for api_key, workflows in item.items():
                    for processed_sources in workflows.values():
                        payloads_by_api_key.setdefault(api_key, []).extend(processed_sources.values())

        ssl_verify = True
        if "localhost" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False
        if "127.0.0.1" in self._settings.api_usage_endpoint_url.lower():
            ssl_verify = False

        for api_key, payloads in payloads_by_api_key.items():
            try:
                response = requests.post(
                    self._settings.api_usage_endpoint_url,
                    json=payloads,
                    verify=ssl_verify,
                    headers={"Authorization": f"Bearer {api_key}"})
            except Exception as exc:
                logger.warning("Failed to send usage - %s", exc)
                # TODO: put back into the queue
                continue
            if response.status_code != 200:
                logger.warning("Failed to send usage - got %s status code (%s)", response.status_code, response.raw)
                # TODO: put back into the queue
                continue

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
