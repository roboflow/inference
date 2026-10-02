"""Aggregation of usage rows per flush window and their delivery."""

import functools
import importlib.metadata
import json
import logging
import numbers
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue
from threading import Event, Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from inference_sdk.config import execution_id
from inference_server import configuration
from inference_server.usage.delivery import (
    STOP_TIMEOUT_S,
    Delivery,
    PendingItem,
    lock_guard,
    pending_item,
)
from inference_server.usage.payload_helpers import (
    APIKey,
    APIKeyHash,
    APIKeyUsage,
    ResourceCategory,
    ResourceID,
    SystemDetails,
    UsagePayload,
    billable_lists_exceed_bound,
    merge_megapixel_buckets,
    merge_resource_details,
    sha256_hash,
    split_billable_lists,
)
from inference_server.usage.queues import RedisQueue, SQLiteQueue

try:
    from streamvision.stream.session import stream_session_id as _stream_session_id_var
except ImportError:
    _stream_session_id_var = None

logger = logging.getLogger(__name__)

SUCCESS_OUTCOME = "success"
ERROR_OUTCOME = "error"
UNKNOWN_ERROR_TYPE = "unknown"
ERROR_TYPE_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
ERROR_KEY = "error"
ERROR_STATUS_CODE_KEY = "error_status_code"
ERROR_TYPE_KEY = "error_type"
BILLABLE_KEY = "billable"
PREVIEW_KEY = "is_preview"
EXTERNAL_SERVICE_NAME = "external"
MAX_AGGREGATED_ROWS = 4096
SYSTEM_INFO_WAIT_S = 5.0
OFFLINE_IP_ADDRESS = "127.0.0.1"
_REPORTED_PACKAGES = {
    "inference_models_version": "inference-models",
    "inference_model_manager_version": "inference-model-manager",
    "roboflow_workflows_version": "roboflow-workflows",
    "streamvision_version": "streamvision",
}


@functools.cache
def _package_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for field, distribution in _REPORTED_PACKAGES.items():
        try:
            versions[field] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[field] = None

    return versions


def _current_stream_session_id() -> Optional[str]:
    if _stream_session_id_var is None:
        return None

    stream_session_id = _stream_session_id_var.get()

    return stream_session_id


def _select_queue(
    *,
    redis_client: Optional[Any],
    sqlite_db_file_path: Optional[Union[str, Path]],
) -> Tuple[Any, bool]:
    serverless = configuration.LAMBDA or configuration.GCP_SERVERLESS
    if serverless and configuration.REDIS_HOST:
        try:
            redis_queue = RedisQueue(redis_client=redis_client)
        except ImportError:
            logger.error("Redis client is not installed, usage is queued in memory")
        else:
            return redis_queue, False
    if serverless or not configuration.TELEMETRY_USE_PERSISTENT_QUEUE:
        memory_queue: "Queue[UsagePayload]" = Queue(
            maxsize=configuration.TELEMETRY_QUEUE_SIZE
        )

        return memory_queue, False

    try:
        sqlite_queue = SQLiteQueue(db_file_path=sqlite_db_file_path)
    except Exception as error:
        logger.debug(
            "Unable to create the persistent usage queue: %s", type(error).__name__
        )
        memory_queue = Queue(maxsize=configuration.TELEMETRY_QUEUE_SIZE)

        return memory_queue, False

    return sqlite_queue, True


class UsageCollector:
    """Aggregates usage rows per flush window and sends them to the platform.

    Rows are accumulated in memory by API key and aggregation key, detached
    every flush interval into a bounded pending list and handed from there to a
    queue and to the platform by ``Delivery``. Nothing runs in the background
    until ``start`` is called.
    """

    def __init__(
        self,
        *,
        redis_client: Optional[Any] = None,
        sqlite_db_file_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Select the queue for the deployment mode.

        Args:
            redis_client: Redis client used by the serverless queue instead of
                one built from the ``REDIS_*`` settings.
            sqlite_db_file_path: File of the persistent queue instead of
                ``usage.db`` in ``MODEL_CACHE_DIR``.
        """
        self._exec_session_id = f"{time.time_ns()}_{uuid4().hex[:4]}"

        self._usage_lock = Lock()
        self._usage: APIKeyUsage = self.empty_usage_dict(
            exec_session_id=self._exec_session_id
        )
        self._rows_count = 0
        self._accepting = True
        self._admission_closed = Event()
        self._ignored_lock = Lock()
        self._ignored_after_stop = 0

        self._api_keys_lock = Lock()
        self._hashed_api_keys: Dict[APIKey, APIKeyHash] = {}

        queue, self._api_keys_hashing_enabled = _select_queue(
            redis_client=redis_client,
            sqlite_db_file_path=sqlite_db_file_path,
        )

        self._system_info_lock = Lock()
        self._system_info_compute_lock = Lock()
        self._system_info: Dict[str, Any] = {}
        self._resolver_started = False
        self._system_info_attempted = Event()

        self._delivery = Delivery(
            queue,
            detach_window=self._detach_window,
            prepare=self._resolve_system_info_in_background,
            host_values=self._host_values,
            resolve_api_keys=self._api_keys_by_hash,
            register_api_key=self._calculate_api_key_hash,
        )

    @property
    def inline_queue_writes(self) -> int:
        """Pending items the recording thread wrote to the queue itself.

        Returns:
            How many times the pending list was full and a caller of
            ``record_usage`` had to write its oldest item to the queue.
        """
        return self._delivery.inline_queue_writes

    @property
    def dropped_rows(self) -> int:
        """Rows given up because the pending list was full and the queue refused."""
        return self._delivery.dropped_rows

    @property
    def dropped_frames(self) -> int:
        """Processed frames of the dropped rows."""
        return self._delivery.dropped_frames

    @property
    def unconfirmed_rows(self) -> int:
        """Rows given up because the queue write may have landed but is unconfirmed."""
        return self._delivery.unconfirmed_rows

    @property
    def unconfirmed_frames(self) -> int:
        """Processed frames of the unconfirmed rows."""
        return self._delivery.unconfirmed_frames

    @property
    def ignored_after_stop(self) -> int:
        """Calls of ``record_usage`` ignored because ``stop`` had returned."""
        return self._ignored_after_stop

    @staticmethod
    def empty_usage_dict(exec_session_id: str) -> APIKeyUsage:
        """Build the empty aggregation of one flush window.

        Args:
            exec_session_id: Execution session written to every new row.

        Returns:
            Mapping of API key hash to aggregation key to row; a missing row is
            created from the row template on first access.
        """
        usage_dict = {
            "timestamp_start": None,
            "timestamp_stop": None,
            "exec_session_id": exec_session_id,
            "hostname": "",
            "ip_address_hash": "",
            "processed_frames": 0,
            "fps": 0,
            "source_duration": 0,
            "category": "",
            "resource_id": "",
            "resource_details": "{}",
            "hosted": bool(configuration.LAMBDA)
            or bool(configuration.DEDICATED_DEPLOYMENT_ID)
            or bool(configuration.GCP_SERVERLESS)
            or bool(configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET),
            "api_key_hash": "",
            "is_gpu_available": False,
            "python_version": sys.version.split()[0],
            "inference_version": configuration.SERVER_VERSION,
            **_package_versions(),
            "enterprise": False,
            "execution_duration": 0,
            "megapixel_buckets": {},
        }
        if configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET:
            usage_dict["roboflow_internal_secret"] = (
                configuration.ROBOFLOW_INTERNAL_SERVICE_SECRET
            )
        if configuration.ROBOFLOW_INTERNAL_SERVICE_NAME:
            usage_dict["roboflow_service_name"] = (
                configuration.ROBOFLOW_INTERNAL_SERVICE_NAME
            )

        return defaultdict(lambda: defaultdict(lambda: {**usage_dict}))

    def _calculate_api_key_hash(self, api_key: APIKey) -> APIKeyHash:
        with self._api_keys_lock:
            api_key_hash = self._hashed_api_keys.get(api_key)
            if not api_key_hash:
                if self._api_keys_hashing_enabled:
                    api_key_hash = sha256_hash(api_key, length=-1)
                else:
                    api_key_hash = api_key
                self._hashed_api_keys[api_key] = api_key_hash
        return api_key_hash

    @staticmethod
    def _calculate_resource_hash(resource_details: Dict[str, Any]) -> str:
        return sha256_hash(json.dumps(resource_details, sort_keys=True))

    @staticmethod
    def _is_billable(resource_details: Optional[Dict[str, Any]]) -> bool:
        if not resource_details:
            return True
        billable = resource_details.get(BILLABLE_KEY)
        return not (
            billable is False
            or (isinstance(billable, str) and billable.lower() == "false")
        )

    @staticmethod
    def _is_preview(resource_details: Optional[Dict[str, Any]]) -> bool:
        if not resource_details:
            return False
        preview = resource_details.get(PREVIEW_KEY)
        return preview is True or (
            isinstance(preview, str) and preview.lower() == "true"
        )

    @staticmethod
    def _normalize_error_type(error_type: Any) -> str:
        if not isinstance(error_type, str):
            return UNKNOWN_ERROR_TYPE
        error_type = error_type.strip()
        if not ERROR_TYPE_PATTERN.fullmatch(error_type):
            return UNKNOWN_ERROR_TYPE
        return error_type

    @staticmethod
    def _normalize_error_status_code(status_code: Any) -> Optional[int]:
        if isinstance(status_code, bool) or not isinstance(
            status_code, numbers.Integral
        ):
            return None
        status_code = int(status_code)
        if not 400 <= status_code <= 599:
            return None
        return status_code

    @classmethod
    def _normalize_error_metadata(
        cls, resource_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        resource_details = dict(resource_details)
        if ERROR_KEY not in resource_details:
            resource_details.pop(ERROR_TYPE_KEY, None)
            resource_details.pop(ERROR_STATUS_CODE_KEY, None)
            return resource_details

        resource_details[ERROR_TYPE_KEY] = cls._normalize_error_type(
            resource_details.get(ERROR_TYPE_KEY)
        )
        error_status_code = cls._normalize_error_status_code(
            resource_details.get(ERROR_STATUS_CODE_KEY)
        )
        if error_status_code is None:
            resource_details.pop(ERROR_STATUS_CODE_KEY, None)
        else:
            resource_details[ERROR_STATUS_CODE_KEY] = error_status_code
        return resource_details

    @classmethod
    def _usage_outcome(
        cls, resource_details: Optional[Dict[str, Any]]
    ) -> Tuple[str, Optional[str], Optional[int]]:
        if not resource_details or ERROR_KEY not in resource_details:
            return SUCCESS_OUTCOME, None, None
        error_type = cls._normalize_error_type(resource_details.get(ERROR_TYPE_KEY))
        error_status_code = cls._normalize_error_status_code(
            resource_details.get(ERROR_STATUS_CODE_KEY)
        )
        return ERROR_OUTCOME, error_type, error_status_code

    @classmethod
    def _usage_key(
        cls,
        category: ResourceCategory,
        resource_id: ResourceID,
        resource_details: Optional[Dict[str, Any]],
        stream_session_id: Optional[str] = None,
    ) -> str:
        billable = str(cls._is_billable(resource_details)).lower()
        outcome, error_type, error_status_code = cls._usage_outcome(resource_details)
        usage_key = f"{category}:{resource_id}:billable={billable}:outcome={outcome}"
        if cls._is_preview(resource_details):
            usage_key = f"{usage_key}:preview=true"
        if outcome == ERROR_OUTCOME:
            usage_key = f"{usage_key}:error_type={error_type}"
            if error_status_code is not None:
                usage_key = f"{usage_key}:error_status_code={error_status_code}"
        if stream_session_id:
            usage_key = f"{usage_key}:{stream_session_id}"
        return usage_key

    @staticmethod
    def system_info(
        ip_address: Optional[str] = None,
        hostname: Optional[str] = None,
        dedicated_deployment_id: Optional[str] = None,
    ) -> SystemDetails:
        """Describe the host the rows are reported from.

        Args:
            ip_address: Address to hash instead of the one of this host.
            hostname: Host name to report instead of the one of this host.
            dedicated_deployment_id: Deployment id instead of
                ``DEDICATED_DEPLOYMENT_ID``.

        Returns:
            ``hostname`` (prefixed by the deployment id, or hashed without one),
            ``ip_address_hash`` and ``is_gpu_available``.
        """
        if not dedicated_deployment_id:
            dedicated_deployment_id = configuration.DEDICATED_DEPLOYMENT_ID
        if not hostname:
            try:
                hostname = socket.gethostname()
            except Exception as error:
                logger.warning("Could not obtain hostname: %s", type(error).__name__)
                hostname = ""
        if dedicated_deployment_id:
            hostname = f"{dedicated_deployment_id}:{hostname}"
        else:
            hostname = sha256_hash(hostname)

        if configuration.LEGACY_OFFLINE_MODE and not ip_address:
            ip_address = "127.0.0.1"
        if not ip_address:
            try:
                ip_address = socket.gethostbyname(socket.gethostname())
            except Exception as error:
                logger.warning("Could not obtain IP address: %s", type(error).__name__)
                s = None
                try:
                    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                    s.connect(("8.8.8.8", 80))
                    ip_address = s.getsockname()[0]
                except Exception:
                    ip_address = socket.gethostbyname("localhost")

                if s:
                    s.close()
        ip_address_hash_hex = sha256_hash(ip_address)

        return {
            "hostname": hostname,
            "ip_address_hash": ip_address_hash_hex,
            "is_gpu_available": False,
        }

    def record_system_info(self, ip_address: Optional[str] = None) -> None:
        """Compute the host description once.

        Args:
            ip_address: Address to hash instead of the one of this host.
        """
        with self._system_info_compute_lock:
            if self._known_system_info():
                return

            system_info = self.system_info(ip_address=ip_address)
            with self._system_info_lock:
                if not self._system_info:
                    self._system_info = system_info

    def _known_system_info(self) -> Dict[str, Any]:
        with self._system_info_lock:
            system_info = dict(self._system_info)

        return system_info

    def _resolve_system_info_in_background(self) -> None:
        try:
            self.record_system_info()
        finally:
            self._system_info_attempted.set()

    def _offline_system_info(self, *, keep: bool) -> Dict[str, Any]:
        offline_system_info = self.system_info(ip_address=OFFLINE_IP_ADDRESS)
        if not keep:
            return offline_system_info
        with self._system_info_lock:
            if not self._system_info:
                self._system_info = offline_system_info
            system_info = dict(self._system_info)

        return system_info

    def _host_values(self, deadline: Optional[float]) -> Dict[str, Any]:
        system_info = self._known_system_info()
        if system_info:
            return system_info
        if not self._resolver_started:
            return self._offline_system_info(keep=False)

        wait_s = SYSTEM_INFO_WAIT_S
        if deadline is not None:
            wait_s = min(wait_s, max(0.0, deadline - time.monotonic()))
        self._system_info_attempted.wait(wait_s)
        system_info = self._known_system_info()
        if not system_info:
            system_info = self._offline_system_info(keep=True)

        return system_info

    def _api_keys_by_hash(self) -> Dict[APIKeyHash, APIKey]:
        with self._api_keys_lock:
            api_keys = {
                api_key_hash: api_key
                for api_key, api_key_hash in self._hashed_api_keys.items()
            }

        return api_keys

    @classmethod
    def _request_details(
        cls,
        resource_details: Any,
        *,
        billable: bool,
        is_preview: bool,
        error_type: Optional[str],
        error_status_code: Optional[int],
    ) -> Dict[str, Any]:
        details = dict(resource_details) if isinstance(resource_details, dict) else {}
        details[BILLABLE_KEY] = billable
        if is_preview or PREVIEW_KEY in details:
            details[PREVIEW_KEY] = is_preview
        if error_type is not None:
            details[ERROR_TYPE_KEY] = error_type
            details.setdefault(ERROR_KEY, cls._normalize_error_type(error_type))
        if error_status_code is not None:
            details[ERROR_STATUS_CODE_KEY] = error_status_code
        normalized_details = cls._normalize_error_metadata(details)

        return normalized_details

    def record_usage(
        self,
        *,
        api_key: APIKey,
        category: str,
        resource_id: str,
        resource_details: Dict[str, Any],
        frames: int,
        execution_duration: float,
        source_duration: float = 0.0,
        fps: float = 0.0,
        billable: bool,
        is_preview: bool = False,
        error_type: Optional[str] = None,
        error_status_code: Optional[int] = None,
        exec_session_id: Optional[str] = None,
        roboflow_service_name: Optional[str] = None,
        roboflow_internal_secret: Optional[str] = None,
        megapixel_buckets: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        """Add one unit of usage to the row of its aggregation key.

        Nothing is recorded in offline mode or without an API key. The row is
        selected by category, resource id, billable flag, outcome, preview flag,
        error type, error status code and stream session.

        Args:
            api_key: API key the usage is reported for.
            category: Row category, such as ``request`` or ``model``.
            resource_id: Resource the usage belongs to; derived from the
                resource details when empty.
            resource_details: Details stored with the row. ``models`` and
                ``custom_python`` lists accumulate over the requests of a row,
                every other key is taken from the latest request.
            frames: Number of processed frames.
            execution_duration: Seconds spent executing.
            source_duration: Seconds of source material; derived from ``frames``
                and ``fps`` when zero.
            fps: Frames per second of the source, zero for images.
            billable: Whether the usage is billable.
            is_preview: Whether the usage comes from a preview run.
            error_type: Class name of the error that failed the request.
            error_status_code: HTTP status of the failure, kept when in 400-599.
            exec_session_id: Execution session of the row; defaults to the
                execution id of the request, then to the one of this collector.
            roboflow_service_name: Internal service the request came from.
            roboflow_internal_secret: Secret presented by that service.
            megapixel_buckets: Per megapixel bucket frame and duration counters.

        Raises:
            ValueError: If ``category`` is empty.
        """
        if self._admission_closed.is_set():
            self._count_ignored()
            return
        if configuration.LEGACY_OFFLINE_MODE:
            return
        if not api_key:
            return
        if not category:
            raise ValueError("Category is compulsory when recording resource details.")

        details = self._request_details(
            resource_details,
            billable=billable,
            is_preview=is_preview,
            error_type=error_type,
            error_status_code=error_status_code,
        )
        try:
            frames = int(frames)
        except Exception:
            frames = 0
        if not source_duration:
            source_duration = frames / fps if fps else 0
        if not resource_id:
            resource_id = self._calculate_resource_hash(details)
        if not exec_session_id:
            exec_session_id = execution_id.get()

        api_key_hash = self._calculate_api_key_hash(api_key)
        stream_session_id = _current_stream_session_id()
        usage_key = self._usage_key(
            category=category,
            resource_id=resource_id,
            resource_details=details,
            stream_session_id=stream_session_id,
        )
        system_info = self._known_system_info()
        details_parts = split_billable_lists(details)
        details = details_parts[0]
        extra_details_json = [json.dumps(extra) for extra in details_parts[1:]]

        while True:
            with self._usage_lock:
                if not self._accepting:
                    self._count_ignored()
                    return

                blocked_rows, source_usage, details = self._row_for_no_lock(
                    api_key_hash, usage_key, details
                )
                if blocked_rows is None:
                    extra_items = self._accumulate_no_lock(
                        source_usage,
                        details,
                        extra_details_json,
                        api_key_hash=api_key_hash,
                        usage_key=usage_key,
                        category=category,
                        resource_id=resource_id,
                        frames=frames,
                        execution_duration=execution_duration,
                        source_duration=source_duration,
                        fps=fps,
                        system_info=system_info,
                        roboflow_service_name=roboflow_service_name,
                        roboflow_internal_secret=roboflow_internal_secret,
                        megapixel_buckets=megapixel_buckets,
                        stream_session_id=stream_session_id,
                        exec_session_id=exec_session_id,
                    )
                    break
            self._delivery.make_room(blocked_rows, write=True)

        for item in extra_items:
            self._delivery.add(item)

    def _count_ignored(self) -> None:
        with self._ignored_lock:
            self._ignored_after_stop += 1

    def _row_for_no_lock(
        self, api_key_hash: APIKeyHash, usage_key: str, details: Dict[str, Any]
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]], Dict[str, Any]]:
        source_usage = self._usage.get(api_key_hash, {}).get(usage_key)
        if source_usage is None and self._rows_count >= MAX_AGGREGATED_ROWS:
            blocked_rows = self._detach_window_no_lock()
            if blocked_rows is not None:
                return blocked_rows, None, details
        if source_usage is not None:
            merged_details = merge_resource_details(
                source_usage["resource_details"], details
            )
            if billable_lists_exceed_bound(merged_details):
                closed_row = pending_item({api_key_hash: {usage_key: source_usage}})
                if not self._delivery.try_add(closed_row):
                    return 1, None, details
                del self._usage[api_key_hash][usage_key]
                self._rows_count -= 1
                source_usage = None
            else:
                details = merged_details
        if source_usage is None:
            source_usage = self._usage[api_key_hash][usage_key]
            self._rows_count += 1

        return None, source_usage, details

    def _accumulate_no_lock(
        self,
        source_usage: Dict[str, Any],
        details: Dict[str, Any],
        extra_details_json: List[str],
        *,
        api_key_hash: APIKeyHash,
        usage_key: str,
        category: str,
        resource_id: str,
        frames: int,
        execution_duration: float,
        source_duration: float,
        fps: float,
        system_info: Dict[str, Any],
        roboflow_service_name: Optional[str],
        roboflow_internal_secret: Optional[str],
        megapixel_buckets: Optional[Dict[str, Dict[str, Any]]],
        stream_session_id: Optional[str],
        exec_session_id: Optional[str],
    ) -> List[PendingItem]:
        if not source_usage["timestamp_start"]:
            source_usage["timestamp_start"] = time.time_ns()
        source_usage["timestamp_stop"] = time.time_ns()
        source_usage["processed_frames"] += frames
        source_usage["source_duration"] += source_duration
        source_usage["fps"] = fps if isinstance(fps, numbers.Number) else 0
        source_usage["category"] = category
        source_usage["resource_id"] = resource_id
        source_usage["resource_details"] = json.dumps(details)
        source_usage["api_key_hash"] = api_key_hash
        if system_info:
            source_usage["hostname"] = system_info["hostname"]
            source_usage["ip_address_hash"] = system_info["ip_address_hash"]
            source_usage["is_gpu_available"] = system_info["is_gpu_available"]
        source_usage["execution_duration"] += execution_duration
        if megapixel_buckets:
            source_usage["megapixel_buckets"] = merge_megapixel_buckets(
                source_usage.get("megapixel_buckets"),
                megapixel_buckets,
            )
        if (
            roboflow_service_name
            and roboflow_service_name != EXTERNAL_SERVICE_NAME
            and roboflow_internal_secret
        ):
            source_usage["roboflow_service_name"] = roboflow_service_name
            source_usage["roboflow_internal_secret"] = roboflow_internal_secret
        if stream_session_id:
            source_usage["stream_session_id"] = stream_session_id
        if exec_session_id:
            source_usage["exec_session_id"] = exec_session_id

        extra_items = []
        for extra_json in extra_details_json:
            extra_row = {
                **source_usage,
                "timestamp_start": source_usage["timestamp_stop"],
                "processed_frames": 0,
                "source_duration": 0,
                "execution_duration": 0,
                "megapixel_buckets": {},
                "resource_details": extra_json,
            }
            extra_items.append(pending_item({api_key_hash: {usage_key: extra_row}}))

        return extra_items

    def _detach_window_no_lock(self) -> Optional[int]:
        window_item = pending_item(self._usage)
        if window_item is None:
            return None
        if not self._delivery.try_add(window_item):
            return window_item.rows

        self._usage = self.empty_usage_dict(exec_session_id=self._exec_session_id)
        self._rows_count = 0

        return None

    def _detach_window(
        self,
        *,
        write: bool = True,
        deadline: Optional[float] = None,
        may_write: Optional[Callable[[], bool]] = None,
        final: bool = False,
    ) -> bool:
        while True:
            with lock_guard(self._usage_lock, deadline) as held:
                if not held:
                    return False
                if final:
                    self._accepting = False
                blocked_rows = self._detach_window_no_lock()
            if blocked_rows is None:
                return True

            made = self._delivery.make_room(
                blocked_rows, write=write, deadline=deadline, may_write=may_write
            )
            if not made:
                return False

    def _enqueue_usage_payload(self) -> None:
        self._detach_window()
        self._delivery.drain_pending()

    def push_usage_payloads(self) -> None:
        """Queue the current window and send everything that is queued."""
        self._enqueue_usage_payload()
        self._delivery.send_queued()

    def flush(self) -> None:
        """Queue the current window and send everything that is queued.

        Blocks for one request per API key and execution session, each limited
        to one second. Rows that are not accepted stay queued or pending.
        """
        self.push_usage_payloads()

    def start(self) -> None:
        """Start the collector and sender threads; a second call does nothing."""
        if self._delivery.started or self._admission_closed.is_set():
            return

        self._resolver_started = True
        self._delivery.start()

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> bool:
        """Stop both threads, then hand over everything and make one send pass.

        Both threads are told to stop and joined until the deadline. From then
        on a thread finishes at most the queue write or request it is inside
        and returns what it holds to the pending list. When both exited, the
        calling thread detaches the current window, writes the pending items to
        the queue and posts what is queued, checking the deadline before every
        queue write and request. When a thread is still inside an operation, the
        calling thread only detaches the window in memory. After ``stop``
        returns, ``record_usage`` ignores its calls; ``flush`` still works. A
        stopped collector cannot be started again.

        Args:
            timeout: Seconds the whole call may take.

        Returns:
            True when both threads exited, nothing is left pending and every
            request of the final pass was started. False otherwise; what was not
            delivered stays pending or queued.
        """
        self._admission_closed.set()
        stopped = self._delivery.stop(timeout)

        return stopped
