import json
import logging
import platform
import re
import socket
import threading
import time
import uuid
from collections import OrderedDict, deque
from itertools import islice
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

import requests
from pydantic import BaseModel

from inference_server import configuration
from inference_server.legacy.entities import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    MultiLabelClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference_server.platform_http import tls_verification_options, wrap_url
from inference_server.server_identity import get_inference_server_id

logger = logging.getLogger(__name__)

MAX_RECORDED_MODELS = 25
MAX_ITEMS_PER_MODEL = 500
MAX_PREDICTIONS_PER_ITEM = 100
MAX_RESPONSES_PER_ITEM = 100
MAX_DETECTIONS_INSPECTED_PER_ITEM = 1000
MAX_STRING_LENGTH = 256
MAX_FULL_ITEM_BYTES = 256 * 1024
MAX_FULL_NODES = 8192
MAX_FULL_STRING_CHARS = 128 * 1024
EXPIRY_INTERVALS = 2
POST_TIMEOUT_S = 10
STOP_JOIN_TIMEOUT_S = 2.0
PREVIOUS_SENDER_JOIN_TIMEOUT_S = POST_TIMEOUT_S + STOP_JOIN_TIMEOUT_S
SENDER_THREAD_NAME = "pingback-sender"

_INCLUDED_REQUEST_FIELDS = {
    "api_key",
    "confidence",
    "model_id",
    "model_type",
    "source",
    "source_info",
}
_CLASS_AND_CONFIDENCE_RESPONSES = (
    ClassificationInferenceResponse,
    ObjectDetectionInferenceResponse,
    InstanceSegmentationInferenceResponse,
)


def _clock() -> float:
    return time.time()


def _bounded(value: Any) -> Any:
    if isinstance(value, str):
        return value[:MAX_STRING_LENGTH]
    if value is None or isinstance(value, (bool, int, float)):
        return value

    return None


class InferenceRecorder:
    def __init__(
        self,
        *,
        max_models: int = MAX_RECORDED_MODELS,
        max_items_per_model: int = MAX_ITEMS_PER_MODEL,
    ) -> None:
        self._max_models = max_models
        self._max_items_per_model = max_items_per_model
        self._lock = threading.Lock()
        self._items: "OrderedDict[str, Deque[Tuple[float, dict]]]" = OrderedDict()
        self.fallback_api_key: Optional[str] = None
        self.failures = 0

    def record(self, model_id: str, item: dict, *, finish_time: float) -> None:
        model_id = _bounded(model_id)
        oldest_kept = finish_time - _expiry_s()
        with self._lock:
            items = self._items.get(model_id)
            if items is None:
                items = deque(maxlen=self._max_items_per_model)
                self._items[model_id] = items
            self._items.move_to_end(model_id)
            while items and items[0][0] < oldest_kept:
                items.popleft()
            items.append((finish_time, item))
            while len(self._items) > self._max_models:
                self._items.popitem(last=False)

    def results(self, *, start: float, stop: float) -> List[dict]:
        oldest_kept = stop - _expiry_s()
        server_id = get_inference_server_id()
        results = []
        with self._lock:
            for model_id in list(self._items):
                items = self._items[model_id]
                while items and items[0][0] < oldest_kept:
                    items.popleft()
                if not items:
                    del self._items[model_id]
                    continue
                results.extend(
                    {
                        "request_time": finish_time,
                        "inference": {**item, "inference_server_id": server_id},
                    }
                    for finish_time, item in items
                    if start <= finish_time <= stop
                )

        return results


RECORDER = InferenceRecorder()


def _expiry_s() -> float:
    return EXPIRY_INTERVALS * configuration.METRICS_INTERVAL


def remember_api_key(api_key: Optional[str], *, monitoring: bool) -> None:
    if monitoring and configuration.METRICS_ENABLED:
        RECORDER.fallback_api_key = _bounded(api_key)


def record_inference(model_id: str, request: Any, response: Any) -> None:
    if getattr(request, "disable_model_monitoring", False):
        return
    if not configuration.METRICS_ENABLED:
        return
    remember_api_key(getattr(request, "api_key", None), monitoring=True)
    if configuration.LEGACY_OFFLINE_MODE or configuration.DISABLE_INFERENCE_CACHE:
        return

    try:
        item = to_cachable_inference_item(request, response)
        RECORDER.record(model_id, item, finish_time=_clock())
    except Exception:
        RECORDER.failures += 1


def to_cachable_inference_item(infer_request: Any, infer_response: Any) -> dict:
    if not configuration.TINY_CACHE:
        full_item = _full_item(infer_request, infer_response)
        if full_item is not None:
            return full_item

    return _condensed_item(infer_request, infer_response)


def _fields_without_image(entity: BaseModel) -> List[Any]:
    return [value for name, value in vars(entity).items() if name != "image"]


def _within_full_thresholds(entities: List[BaseModel]) -> bool:
    nodes = MAX_FULL_NODES
    characters = MAX_FULL_STRING_CHARS
    stack: List[Any] = []
    for entity in entities:
        stack.extend(_fields_without_image(entity))
    while stack:
        value = stack.pop()
        nodes -= 1
        if nodes < 0:
            return False
        if value is None or isinstance(value, (bool, int, float)):
            continue
        if isinstance(value, (str, bytes)):
            characters -= len(value)
            if characters < 0:
                return False
        elif isinstance(value, BaseModel):
            stack.extend(vars(value).values())
        elif isinstance(value, (list, tuple)):
            stack.extend(islice(value, nodes + 1))
        elif isinstance(value, dict):
            for key, element in islice(value.items(), nodes + 1):
                stack.append(key)
                stack.append(element)
        else:
            return False

    return True


def _full_item(infer_request: Any, infer_response: Any) -> Optional[dict]:
    responses = infer_response if isinstance(infer_response, list) else [infer_response]
    if len(responses) > MAX_FULL_NODES:
        return None
    entities = [infer_request, *responses]
    if not all(isinstance(entity, BaseModel) for entity in entities):
        return None
    if not _within_full_thresholds(entities):
        return None

    dumped = [
        entity.model_dump(mode="json", by_alias=True, exclude={"image"})
        for entity in entities
    ]
    item = {
        "inference_id": infer_request.id,
        "inference_server_version": configuration.SERVER_VERSION,
        "request": dumped[0],
        "response": dumped[1:] if isinstance(infer_response, list) else dumped[1],
    }
    try:
        size = len(json.dumps(item, allow_nan=False))
    except ValueError:
        return None
    if size > MAX_FULL_ITEM_BYTES:
        return None

    return item


def _condensed_item(infer_request: Any, infer_response: Any) -> dict:
    item = {
        "inference_id": _bounded(infer_request.id),
        "inference_server_version": configuration.SERVER_VERSION,
        "request": {
            name: _bounded(value)
            for name, value in infer_request.model_dump(
                include=_INCLUDED_REQUEST_FIELDS
            ).items()
        },
        "response": build_condensed_response(infer_response),
    }

    return item


class _Budget:
    __slots__ = ("predictions", "detections")

    def __init__(self) -> None:
        self.predictions = MAX_PREDICTIONS_PER_ITEM
        self.detections = MAX_DETECTIONS_INSPECTED_PER_ITEM


_END = object()


def _inspected(source: Any, budget: _Budget) -> Iterator[Any]:
    iterator = iter(source)
    while budget.detections > 0:
        element = next(iterator, _END)
        if element is _END:
            return
        budget.detections -= 1
        yield element


def build_condensed_response(responses: Any) -> List[dict]:
    if not isinstance(responses, list):
        responses = [responses]

    formatted_responses = []
    budget = _Budget()
    for response in islice(responses, MAX_RESPONSES_PER_ITEM):
        if budget.predictions <= 0 or budget.detections <= 0:
            break
        if not getattr(response, "predictions", None):
            continue
        try:
            entries = _condensed_predictions(response, budget)
            if entries is None:
                continue
            predictions = list(islice(entries, budget.predictions))
            formatted_responses.append(
                {"predictions": predictions, "time": _bounded(response.time)}
            )
            budget.predictions -= len(predictions)
        except Exception:
            RECORDER.failures += 1

    return formatted_responses


def _condensed_predictions(response: Any, budget: _Budget) -> Optional[Iterator[dict]]:
    if isinstance(response, MultiLabelClassificationInferenceResponse):
        entries = (
            {"class": _bounded(class_name), "confidence": prediction.confidence}
            for class_name, prediction in _inspected(
                response.predictions.items(), budget
            )
        )

        return entries

    if isinstance(response, KeypointsDetectionInferenceResponse):
        predictions = (
            keypoint
            for prediction in _inspected(response.predictions, budget)
            for keypoint in prediction.keypoints
        )
    elif isinstance(response, _CLASS_AND_CONFIDENCE_RESPONSES):
        predictions = _inspected(response.predictions, budget)
    else:
        return None

    entries = (
        {"class": _bounded(prediction.class_name), "confidence": prediction.confidence}
        for prediction in predictions
    )

    return entries


def get_system_info() -> Dict[str, str]:
    info: Dict[str, str] = {}
    try:
        info["platform"] = platform.system()
        info["platform_release"] = platform.release()
        info["platform_version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip_address"] = socket.gethostbyname(socket.gethostname())
        info["mac_address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
    except Exception as error:
        logger.warning("Could not collect system information: %s", type(error).__name__)

    return info


def _device_id() -> str:
    if configuration.DEVICE_ID is not None:
        return configuration.DEVICE_ID

    hostname = platform.node()

    return hostname


class PingbackSender:
    def __init__(
        self,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wait: Optional[Callable[[float], bool]] = None,
    ) -> None:
        self._stop_requested = threading.Event()
        self._monotonic = monotonic
        self._wait = wait if wait is not None else self._stop_requested.wait
        self._thread: Optional[threading.Thread] = None
        self._environment_info: Optional[dict] = None
        self._created_at = str(int(_clock()))

    def _environment(self) -> dict:
        if self._environment_info is None:
            self._environment_info = {
                "api_key": configuration.METRICS_API_KEY,
                "timestamp": self._created_at,
                "device_id": _device_id(),
                "inference_server_version": configuration.SERVER_VERSION,
                "tags": configuration.TAGS,
                **get_system_info(),
            }

        return self._environment_info

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run, name=SENDER_THREAD_NAME, daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = STOP_JOIN_TIMEOUT_S) -> None:
        self._stop_requested.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_running(self) -> bool:
        return (
            self._thread is not None
            and self._thread.is_alive()
            and not self._stop_requested.is_set()
        )

    def _run(self) -> None:
        interval = configuration.METRICS_INTERVAL
        start = self._monotonic()
        tick = 1
        while not self._wait(max(0.0, start + tick * interval - self._monotonic())):
            self.post()
            missed = int((self._monotonic() - start) // interval) if interval > 0 else 0
            tick = max(tick + 1, missed + 1)

    def build_payload(self) -> dict:
        payload = dict(self._environment())
        payload["inference_server_id"] = get_inference_server_id()
        if RECORDER.fallback_api_key and not payload.get("api_key"):
            payload["api_key"] = RECORDER.fallback_api_key
        now = _clock()
        payload["inference_results"] = RECORDER.results(
            start=now - configuration.METRICS_INTERVAL, stop=now
        )

        return payload

    def post(self) -> None:
        if configuration.LEGACY_OFFLINE_MODE:
            return

        try:
            response = requests.post(
                wrap_url(configuration.METRICS_URL),
                json=self.build_payload(),
                timeout=POST_TIMEOUT_S,
                **tls_verification_options(),
            )
            status_code = response.status_code
        except Exception as error:
            logger.warning(
                "Sending metrics to Roboflow failed: %s", type(error).__name__
            )
            return

        if status_code >= 400:
            logger.debug(
                "Sending metrics to Roboflow failed with HTTP status %s", status_code
            )


_SENDER: Optional[PingbackSender] = None
_SENDER_LOCK = threading.Lock()


def start_sender() -> Optional[PingbackSender]:
    global _SENDER

    if not configuration.METRICS_ENABLED:
        return None

    with _SENDER_LOCK:
        previous = _SENDER
        if previous is not None:
            if previous.is_running():
                return previous
            thread = previous._thread
            if thread is not None and thread.is_alive():
                thread.join(timeout=PREVIOUS_SENDER_JOIN_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning("Previous metrics sender is still running")
                    return None

        sender = PingbackSender()
        sender.start()
        _SENDER = sender

    return sender
