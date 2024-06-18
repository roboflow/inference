import atexit
from collections import defaultdict
import hashlib
import json
import importlib_metadata
import mimetypes
from queue import Queue
import socket
import sys
from threading import Event, Lock, Thread
import time
import torch
from typing import Any, DefaultDict, Dict, Optional
from uuid import uuid4

from inference.core.env import API_KEY
from inference.core.logger import logger

from .config import get_telemetry_settings, TelemetrySettings


Source = str
UsagePayload = DefaultDict[Source, Any]
WorkflowID = str
WorkflowUsage = DefaultDict[WorkflowID, UsagePayload]
ApiKey = str
APIKeyUsage = DefaultDict[ApiKey, WorkflowUsage]


class UsageCollector:
    def __init__(self):
        self._lock = Lock()

        self._settings: TelemetrySettings = get_telemetry_settings()
        self._usage: APIKeyUsage = self._get_empty_usage_dict()

        # TODO: use persistent queue, i.e. https://pypi.org/project/persist-queue/
        self._queue = Queue(maxsize=self._settings.queue_size)

        _ip_address: str = socket.gethostbyname(
            socket.gethostname()
        )
        _ip_address_hash = hashlib.sha256()
        _ip_address_hash.update(_ip_address.encode())
        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"
        self._ip_address_hash = _ip_address_hash.hexdigest()
        self._gpu_available: bool = torch.cuda.is_available()
        try:
            self._inference_version: str = importlib_metadata.version("inference")
        except importlib_metadata.PackageNotFoundError:
            self._inference_version: str = "dev"

        self._terminate_scheduler = Event()
        self._scheduler_thread = Thread(target=self._scheduler, daemon=True)
        self._scheduler_thread.start()

        atexit.register(self._cleanup)

    def _scheduler(self):
        while True:
            if self._terminate_scheduler.wait(self._settings.flush_interval):
                break
            self._send_usage()
        logger.debug("Terminating %s", self.__class__.__name__)
        self._send_usage()

    def _send_usage(self):
        with self._lock:
            if not self._usage:
                return
            # TODO: aggregate last element of the queue if maxsize is reached
            self._queue.put(self._usage)
            self._usage = self._get_empty_usage_dict()
            logger.debug("Queue: %s", [e for e in self._queue.queue])

    @staticmethod
    def _guess_source_type(source) -> str:
        mime_type, _ = mimetypes.guess_type(source)
        source_type = "unknown"
        if mime_type and mime_type.startswith("video"):
            source_type = "video"
        elif mime_type and mime_type.startswith("image"):
            source_type = "image"
        elif mime_type is None:
            stream_schemes = ["rtsp", "rtmp"]
            if any(source.lower().startswith(s) for s in stream_schemes):
                source_type = "stream"
        return source_type

    @staticmethod
    def _calculate_workflow_hash(workflow: Dict[str, Any]) -> str:
        workflow_hash = hashlib.sha256()
        workflow_hash.update(
            json.dumps(workflow, sort_keys=True).encode()
        )
        return workflow_hash.hexdigest()

    def _get_empty_usage_dict(self) -> APIKeyUsage:
        return defaultdict(  # API key
            lambda: defaultdict(  # workflow ID
                lambda: defaultdict(  # source hash
                    lambda: {
                        "timestamp": None,
                        "exec_session_id": self._exec_session_id,
                        "source_hash": None,
                        "source_type": None,
                        "processed_frames": 0,
                        "fps": 0,
                        "source_duration": 0,
                        "with_workflow": False,
                    }
                )
            )
        )

    def record_usage(self, source: str, frames: int, api_key: Optional[str] = None, workflow: Optional[Dict[str, Any]] = None, workflow_id: Optional[str] = None, fps: Optional[float] = 0) -> DefaultDict[str, Any]:
        source_type = UsageCollector._guess_source_type(source=source)
        source_hash = hashlib.sha256()
        source_hash.update(source.encode())
        if not api_key:
            api_key = API_KEY
        if workflow_id is None:
            workflow_id = self._calculate_workflow_hash(workflow) if workflow else "No Workflow"
        with self._lock:
            source_usage = self._usage[api_key][workflow_id][source_hash.hexdigest()]  # TODO: same source can be processed multiple times
            source_usage["timestamp"] = time.time()
            source_usage["source_hash"] = source_hash
            source_usage["source_type"] = source_type
            source_usage["processed_frames"] += frames
            source_usage["fps"] = fps
            source_usage["source_duration"] += frames / fps if fps else 0
            source_usage["with_workflow"] = workflow is not None

    def _get_system_info(self, api_key: Optional[str] = None, ip_address_hash: Optional[str] = None):
        if not ip_address_hash:
            ip_address_hash = self._ip_address_hash
        if not api_key:
            api_key = API_KEY
        return {
            "timestamp": time.time(),
            "exec_session_id": self._exec_session_id,
            "ip_address_hash": ip_address_hash,
            "api_key": api_key,
            "is_gpu_available": self._gpu_available,
            "python_version": sys.version.split()[0],
            "inference_version": self._inference_version,
        }

    def record_execution_details(self, workflow, api_key: Optional[str] = None, workflow_id: Optional[str] = None):
        if not api_key:
            api_key = API_KEY
        if workflow_id is None:
            workflow_id = self._calculate_workflow_hash(workflow)
        with self._lock:
            self._queue.put({
                api_key: {
                    workflow_id: workflow,
                }
            })
            self._queue.put(self._get_system_info())

    def _cleanup(self):
        self._terminate_scheduler.set()
        self._scheduler_thread.join()
