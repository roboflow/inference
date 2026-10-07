from __future__ import annotations

import asyncio
import contextvars
import logging
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias

from inference_models.errors import ModelInputError
from inference_server import platform_http, telemetry
from inference_server.configuration import (
    ALLOW_URL_INPUT,
    API_BASE_URL,
    INFER_TIMEOUT_S,
    LEGACY_LOAD_POLL_INTERVAL_S,
    LEGACY_LOAD_TIMEOUT_S,
    LEGACY_OFFLINE_MODE,
    LEGACY_ROUTE_METADATA_TTL_S,
    MODEL_STAT_CACHE_SIZE,
    MODEL_STAT_CACHE_TTL_S,
)
from inference_server.errors import PayloadTooLargeError
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.fanout import gather_bounded
from inference_server.framework.model_stat import (
    ModelStat,
    _TtlLruCache,
    stat_model_details_while_checking_auth,
)
from inference_server.gateway import ModelManagerGateway, ReloadAfterEvictionError
from inference_server.legacy.common import ImagePayload, fetch_url_images
from inference_server.legacy.entities import ResolvedModel
from inference_server.legacy.errors import (
    MODEL_ACCESS_ERROR_MESSAGES,
    MODEL_PACKAGE_BROKEN_MESSAGE,
    NOT_FOUND_MESSAGE,
    REGISTRY_REQUEST_FAILED_MESSAGE,
    REGISTRY_UNREACHABLE_MESSAGE,
    SERVICE_MISCONFIGURATION_MESSAGE,
    UNAUTHORIZED_MESSAGE,
    ImageFetchError,
    LegacyHTTPError,
    ModelNotReadyError,
)
from inference_server.legacy.load_failures import load_failure_error
from inference_server.legacy.telemetry_recording import (
    RECORDED_LOAD_EVENTS,
    record_telemetry,
)
from inference_server.middlewares.model_load import (
    MODEL_LOAD_EVENTS,
    REQUESTED_MODEL_ID,
    record_model_load,
    set_requested_model_id,
)
from inference_server.prometheus import measure_inference
from inference_server.routing import (
    has_capability_marker,
    parse_registration_key,
    plain_model_id,
    routing_key,
)
from inference_server.usage.rows import (
    MEGAPIXEL_BUCKET_UNKNOWN,
    USAGE_SCOPE,
    add_megapixel_bucket,
    bound_scope,
    execution_duration,
    megapixel_bucket,
    record_model_usage,
)

logger = logging.getLogger(__name__)

_ERR_NOT_LOADED = 6
_CURRENT_REQUEST: contextvars.ContextVar[
    Optional[dict[tuple[str, str], tuple[Route, str, str, Optional[str]]]]
] = contextvars.ContextVar("legacy_current_request", default=None)
_USAGE_CALL: contextvars.ContextVar[Optional["_UsageCall"]] = contextvars.ContextVar(
    "legacy_usage_call", default=None
)
_SYNC_TIMEOUT_MARGIN_S = 30
_MAX_PENDING_REQUEST_KEYS = 256
_MAX_PENDING_VALUES_PER_KEY = 64

STUB_VERSION_ID = "0"
STUB_MODEL_ARCHITECTURE = "stub"
MISSING_API_KEY_MESSAGE = (
    "Required Roboflow API key is missing. Visit "
    "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key "
    "to learn how to retrieve one."
)
_STUB_TASK_TYPES = frozenset(
    [
        "classification",
        "object-detection",
        "instance-segmentation",
        "keypoint-detection",
    ]
)
_DEFAULT_PROJECT_TASK_TYPE = "object-detection"
_PLATFORM_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_stub_stats: _TtlLruCache = _TtlLruCache(MODEL_STAT_CACHE_SIZE, MODEL_STAT_CACHE_TTL_S)


def _remember_request(
    route: "Route", model_id_as_requested: str, path: str, alias: Optional[str]
) -> None:
    requests = dict(_CURRENT_REQUEST.get() or {})
    requests[(route.registry_id, model_id_as_requested)] = (
        route,
        model_id_as_requested,
        path,
        alias,
    )
    _CURRENT_REQUEST.set(requests)


@dataclass
class _UsageCall:
    route: "Route"
    api_key: Optional[str]
    expected_images: Optional[list[Optional[ImagePayload]]]
    fragment_images: list[ImagePayload] = field(default_factory=list)
    image_durations: list[Optional[float]] = field(default_factory=list)
    text_durations: list[Optional[float]] = field(default_factory=list)

    def add(
        self,
        images: list[Optional[ImagePayload]],
        model_durations: Optional[list[Optional[float]]],
    ) -> None:
        if model_durations is None:
            model_durations = [None] * len(images)
        for image, duration in zip(images, model_durations):
            if image is None:
                self.text_durations.append(duration)
                continue
            self.fragment_images.append(image)
            self.image_durations.append(duration)

    def images(self) -> list[Optional[ImagePayload]]:
        images = self.expected_images
        if images is None:
            images = self.fragment_images
        combined = [*images, *([None] * len(self.text_durations))]

        return combined

    def model_durations(self) -> Optional[list[Optional[float]]]:
        images = self.expected_images
        if images is None:
            images = self.fragment_images
        present = [image for image in images if image is not None]
        if len(self.image_durations) != len(present):
            return None
        durations = [*self.image_durations, *self.text_durations]
        if any(duration is None for duration in durations):
            return None

        return durations


@contextmanager
def usage_call(
    route: "Route",
    api_key: Optional[str],
    *,
    images: Optional[list[Optional[ImagePayload]]] = None,
) -> Iterator[None]:
    """Record the bridge calls made inside as one ``model`` row increment.

    Calls made with ``record=False`` inside the block are fragments of one
    legacy model call: their model-side durations are combined and recorded
    once when the block exits, with the serverless floor applied to the
    combined duration. A block that raises or is cancelled is recorded with
    the error and the time elapsed until then, and the frames of the whole
    call.

    Args:
        route: Route of the model the fragments call.
        api_key: Key the model is called with.
        images: Images of the whole call, counted whatever fragments ran;
            when None the frames are the images the fragments called with.

    Yields:
        Nothing.
    """
    call = _UsageCall(route=route, api_key=api_key, expected_images=images)
    token = _USAGE_CALL.set(call)
    started = time.perf_counter()
    try:
        yield
    except BaseException as error:
        record_telemetry(
            _record_model_invocation,
            route,
            call.images(),
            api_key,
            time.perf_counter() - started,
            None,
            error,
        )
        raise
    else:
        record_telemetry(
            _record_model_invocation,
            route,
            call.images(),
            api_key,
            time.perf_counter() - started,
            call.model_durations(),
            None,
        )
    finally:
        _USAGE_CALL.reset(token)


@dataclass
class Route:
    model_id: str
    registry_id: str
    task_type: str
    action: str
    actions: set[str] = field(default_factory=set)
    class_names: Optional[list[str]] = None
    key_points_classes: Optional[list[list[str]]] = None
    video_sampling: Optional[dict] = None
    model_class_name: Optional[str] = None
    model_mro_names: list[str] = field(default_factory=list)
    class_colors: Optional[dict] = None
    requested_at: dict[str, float] = field(default_factory=dict)
    request_paths_by_id: dict[str, dict[str, float]] = field(default_factory=dict)
    request_aliases_by_id: dict[str, dict[str, float]] = field(default_factory=dict)
    loaded_monotonic: Optional[float] = None
    input_height: Optional[int] = None
    input_width: Optional[int] = None
    vram_bytes: Optional[int] = None
    resolved_model: Optional[dict] = None
    metadata_ts: float = 0.0
    model_architecture: Optional[str] = None
    model_variant: Optional[str] = None
    is_stub: bool = False
    stream_pipeline_depth: int = 1


_CORE_MODEL_TASK_TYPES: dict[str, tuple[str, str]] = {
    "clip": ("embedding", "embed_images"),
    "perception_encoder": ("embedding", "embed_images"),
    "sam": ("interactive-instance-segmentation", "embed"),
    "sam2": ("interactive-instance-segmentation", "embed"),
    "sam3": ("interactive-instance-segmentation", "embed"),
    "doctr": ("structured-ocr", "infer"),
    "easy_ocr": ("structured-ocr", "infer"),
    "trocr": ("text-only-ocr", "infer"),
    "owlv2": ("object-detection", "infer"),
    "yolo_world": ("open-vocabulary-object-detection", "infer"),
    "grounding_dino": ("open-vocabulary-object-detection", "infer"),
    "depth-anything-v2": ("depth-estimation", "infer"),
    "depth-anything-v3": ("depth-estimation", "infer"),
    "moondream2": ("vlm", "prompt"),
    "smolvlm2": ("vlm", "prompt"),
}

_CORE_MODEL_ARCHITECTURES = {
    "grounding_dino": "grounding-dino",
    "yolo_world": "yolo-world",
    "smolvlm2": "smolvlm-2.2b-instruct",
}

_REGISTRY_ID_ALIASES = {"perception_encoder": "perception-encoder"}

_TASK_TYPE_BY_MRO = {
    "ObjectDetectionModel": "object-detection",
    "OWLv2HF": "open-vocabulary-object-detection",
    "RFDetrForObjectDetectionTorch": "object-detection",
    "RFDetrForObjectDetectionONNX": "object-detection",
    "RFDetrForObjectDetectionTRT": "object-detection",
    "YOLO26ForObjectDetectionOnnx": "object-detection",
    "YOLO26ForObjectDetectionTorchScript": "object-detection",
    "YOLO26ForObjectDetectionTRT": "object-detection",
    "YOLOv10ForObjectDetectionOnnx": "object-detection",
    "YOLOv10ForObjectDetectionTRT": "object-detection",
    "PPOCRv6DetectionOnnx": "object-detection",
    "RoboflowInstantHF": "object-detection",
    "PassthroughModel": "passthrough",
    "OpenVocabularyObjectDetectionModel": "open-vocabulary-object-detection",
    "GroundingDinoForObjectDetectionTorch": "open-vocabulary-object-detection",
    "InstanceSegmentationModel": "instance-segmentation",
    "YOLOv5ForInstanceSegmentationOnnx": "instance-segmentation",
    "YOLOv5ForInstanceSegmentationTRT": "instance-segmentation",
    "YOLOv7ForInstanceSegmentationOnnx": "instance-segmentation",
    "YOLOv7ForInstanceSegmentationTRT": "instance-segmentation",
    "YOLOACTForInstanceSegmentationOnnx": "instance-segmentation",
    "YOLOACTForInstanceSegmentationTRT": "instance-segmentation",
    "RFDetrForInstanceSegmentationTorch": "instance-segmentation",
    "RFDetrForInstanceSegmentationOnnx": "instance-segmentation",
    "RFDetrForInstanceSegmentationTRT": "instance-segmentation",
    "YOLO26ForInstanceSegmentationOnnx": "instance-segmentation",
    "YOLO26ForInstanceSegmentationTorchScript": "instance-segmentation",
    "YOLO26ForInstanceSegmentationTRT": "instance-segmentation",
    "SAM2ForStream": "instance-segmentation",
    "KeyPointsDetectionModel": "keypoint-detection",
    "RFDetrForKeyPointsONNX": "keypoint-detection",
    "YOLO26ForKeyPointsDetectionOnnx": "keypoint-detection",
    "YOLO26ForKeyPointsDetectionTorchScript": "keypoint-detection",
    "YOLO26ForKeyPointsDetectionTRT": "keypoint-detection",
    "ClassificationModel": "classification",
    "MultiLabelClassificationModel": "multi-label-classification",
    "SemanticSegmentationModel": "semantic-segmentation",
    "DepthEstimationModel": "depth-estimation",
    "TextImageEmbeddingModel": "embedding",
    "StructuredOCRModel": "structured-ocr",
    "EasyOCRTorch": "structured-ocr",
    "PPOCRv6StructuredOCR": "structured-ocr",
    "TextOnlyOCRModel": "text-only-ocr",
    "L2CSNetOnnx": "gaze-detection",
    "PaliGemmaHF": "vlm",
    "Gemma4HF": "vlm",
    "Qwen25VLHF": "vlm",
    "Qwen3VLHF": "vlm",
    "Qwen35HF": "vlm",
    "Qwen3VLVLLMProxy": "vlm",
    "Qwen35VLLMProxy": "vlm",
    "Qwen38VLLMProxy": "vlm",
    "SmolVLMHF": "vlm",
    "Cosmos3EdgeReasoner": "vlm",
    "Florence2HF": "vlm",
    "MoonDream2HF": "vlm",
    "GlmOcrHF": "vlm",
    "SAMTorch": "interactive-instance-segmentation",
    "SAM2Torch": "interactive-instance-segmentation",
    "SAM3Torch": "interactive-instance-segmentation",
    "ActionRecognitionModel": "action-recognition",
    "Cosmos3EdgeActionRecognition": "action-recognition",
}

_NO_HTTP_ROUTE = frozenset(["SAM2ForStream"])

_DEFAULT_ACTION_BY_TASK_TYPE = {
    "vlm": "prompt",
    "embedding": "embed_images",
    "interactive-instance-segmentation": "embed",
}


def registry_id_for(model_id: str) -> str:
    resolved = resolve_roboflow_model_alias(model_id)
    dataset, separator, version = resolved.partition("/")
    alias = _REGISTRY_ID_ALIASES.get(dataset)
    if alias is None:
        return resolved
    return f"{alias}{separator}{version}"


def _plain_registry_id(model_id: str) -> str:
    if has_capability_marker(model_id):
        raise LegacyHTTPError(404, NOT_FOUND_MESSAGE)
    return registry_id_for(model_id)


def request_alias_for(model_id: str) -> Optional[str]:
    legacy_model_id = resolve_roboflow_model_alias(model_id)
    if legacy_model_id == model_id:
        return None

    return legacy_model_id


def requested_model_id_for(registry_id: str) -> str:
    model_id = plain_model_id(registry_id)
    requested = REQUESTED_MODEL_ID.get()
    if requested is not None and requested[0] == model_id:
        return requested[1]

    return model_id


def resolved_model_for(route: Route) -> ResolvedModel:
    if route.resolved_model:
        return ResolvedModel(**route.resolved_model)
    return ResolvedModel(model_id=route.registry_id)


class LoopBridge:
    """Run a coroutine on the server loop from any worker thread; raises on the loop thread."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def run(self, coro, timeout: float) -> Any:
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop is self._loop:
            coro.close()
            raise RuntimeError("LoopBridge used on the event loop thread")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)


class LegacyModelBridge:
    def __init__(self, gateway: Any) -> None:
        self.gateway = gateway
        self.accepts_ndarray = isinstance(gateway, ModelManagerGateway)
        self._routes: dict[str, Route] = {}
        self._loaded_ids: set[str] = set()
        self._preloaded_ids: dict[str, dict[str, float]] = {}
        self._pending_requests: dict[str, tuple[set[str], set[str]]] = {}

    async def resolve(
        self,
        model_id: str,
        api_key: Optional[str],
        *,
        row_key: Optional[str] = None,
        path: str = "",
        alias: Optional[str] = None,
        instance: str = "",
    ) -> Route:
        model_registry_id = _plain_registry_id(model_id)
        registry_id = routing_key(model_registry_id, instance)
        set_requested_model_id(model_registry_id, requested_model_id=model_id)
        if model_registry_id.partition("/")[2] == STUB_VERSION_ID:
            stub_route = await _resolve_stub(model_id, model_registry_id, api_key)

            return stub_route

        stat: Optional[ModelStat] = None
        if not LEGACY_OFFLINE_MODE:
            try:
                stat = await self._stat(model_id, model_registry_id, api_key)
            except (Exception, asyncio.CancelledError):
                self._hold_pending_request(row_key, path, alias)
                raise
        route = self._routes.get(registry_id)
        if route is None:
            route = Route(
                model_id=model_id,
                registry_id=registry_id,
                task_type="unknown",
                action="infer",
            )
        _apply_stat(route, stat)
        try:
            if LEGACY_OFFLINE_MODE:
                try:
                    await self.ensure_loaded(route, api_key)
                except LegacyHTTPError as error:
                    raise LegacyHTTPError(
                        404, f"Model {model_id} not available offline"
                    ) from error
            else:
                await self.ensure_loaded(route, api_key)
        except (Exception, asyncio.CancelledError):
            self._hold_pending_request(row_key, path, alias)
            raise
        route = self._adopt_canonical(route, stat)
        if self._metadata_expired(route) or route.registry_id not in self._loaded_ids:
            await self._refresh_metadata(route)
        route = self._adopt_canonical(route, stat)
        if LEGACY_OFFLINE_MODE and stat is None:
            route.task_type = _task_type_from_mro(route.model_mro_names)
            route.action = _DEFAULT_ACTION_BY_TASK_TYPE.get(route.task_type, "infer")
        self._routes[registry_id] = route
        self._routes[routing_key(model_id, instance)] = route
        if route.model_class_name in _NO_HTTP_ROUTE:
            raise LegacyHTTPError(
                400,
                f"Model {model_id!r} is a streaming model without an HTTP "
                "inference route.",
            )
        return route

    async def stat_task_type(
        self, model_id: str, api_key: Optional[str]
    ) -> Optional[str]:
        """Report a model's task type without loading it.

        Args:
            model_id: Model id as requested, alias included.
            api_key: Credential the registry checks access with.

        Returns:
            The registry's task type, or None offline, where only a load can
            tell.
        """
        registry_id = _plain_registry_id(model_id)
        if registry_id.partition("/")[2] == STUB_VERSION_ID:
            stat = await _stat_stub(registry_id, api_key)
        elif LEGACY_OFFLINE_MODE:
            return None
        else:
            stat = await self._stat(model_id, registry_id, api_key)

        return stat.task_type

    def _adopt_canonical(self, route: Route, stat: Optional[ModelStat]) -> Route:
        canonical = self._routes.get(route.registry_id)
        if canonical is None or canonical is route:
            return route
        _apply_stat(canonical, stat)
        return canonical

    async def ensure_loaded(self, route: Route, api_key: Optional[str]) -> None:
        try:
            await self._ensure_loaded(route, api_key)
        finally:
            self._record_cold_starts(route.registry_id)
            self._refresh_current_request(route.registry_id)

    def _record_cold_starts(self, registry_id: str) -> None:
        events = MODEL_LOAD_EVENTS.get()
        recorded = RECORDED_LOAD_EVENTS.get()
        if events is None or recorded is None:
            return

        model_id = requested_model_id_for(registry_id)
        for position, (event_model_id, cold_start, load_time_s) in enumerate(events):
            if not cold_start or event_model_id != model_id or position in recorded:
                continue
            recorded.add(position)
            record_telemetry(telemetry.record_model_loaded, model_id, load_time_s)

    async def _ensure_loaded(self, route: Route, api_key: Optional[str]) -> None:
        record_model_load(
            plain_model_id(route.registry_id), cold_start=False, load_time_s=0.0
        )
        deadline = time.monotonic() + LEGACY_LOAD_TIMEOUT_S
        while True:
            result = await self.gateway.ensure_loaded(
                route.registry_id, "", api_key or "", ""
            )
            state = result[0] if result else "error"
            if state == "model_ready":
                return
            if state == "error":
                code = result[1] if len(result) > 1 else None
                if code == _ERR_NOT_LOADED:
                    raise _not_ready_error()
                raise _load_error(result)
            if time.monotonic() >= deadline:
                raise _not_ready_error()
            await asyncio.sleep(LEGACY_LOAD_POLL_INTERVAL_S)
            failure = self._last_load_failure(route.registry_id)
            if failure is not None:
                raise _load_error(failure)

    def _last_load_failure(self, registry_id: str) -> Optional[tuple]:
        last_load_failure = getattr(self.gateway, "last_load_failure", None)
        if last_load_failure is None:
            return None

        failure = last_load_failure(registry_id)

        return failure

    async def load(
        self,
        model_id: str,
        api_key: Optional[str],
        *,
        row_key: Optional[str] = None,
        path: str = "",
        alias: Optional[str] = None,
    ) -> Route:
        route = await self.resolve(
            model_id, api_key, row_key=row_key, path=path, alias=alias
        )
        try:
            try:
                result = await self.gateway.load(
                    route.registry_id,
                    api_key or "",
                    timeout_s=LEGACY_LOAD_TIMEOUT_S,
                    pinned=False,
                )
            except asyncio.TimeoutError as error:
                raise _not_ready_error() from error
            state = result[0] if result else "error"
            if state != "ok":
                raise _load_error(result)
        except (Exception, asyncio.CancelledError):
            self._hold_pending_request(row_key, path, alias)
            raise
        finally:
            self._record_cold_starts(route.registry_id)
        return route

    async def infer(
        self,
        route: Route,
        api_key: Optional[str],
        action: str,
        images: list[Optional[ImagePayload]],
        params: dict,
        *,
        model_monitoring: bool = True,
        record: bool = True,
    ) -> list[Any]:
        try:
            results, _model_durations = await self._infer(
                route,
                api_key,
                action,
                images,
                params,
                model_monitoring=model_monitoring,
                record=record,
            )
        finally:
            self._record_cold_starts(route.registry_id)
            self._refresh_current_request(route.registry_id)
        return results

    async def _infer(
        self,
        route: Route,
        api_key: Optional[str],
        action: str,
        images: list[Optional[ImagePayload]],
        params: dict,
        *,
        model_monitoring: bool,
        record: bool,
    ) -> tuple[list[Any], list[Optional[float]]]:
        await self.ensure_loaded(route, api_key)
        started = time.perf_counter()
        try:
            with measure_inference(
                route.registry_id,
                responses=len(images),
                monitoring=record and model_monitoring,
            ):
                try:
                    outcomes = await gather_bounded(
                        *(
                            self._infer_one(
                                route.registry_id,
                                image.data if image is not None else None,
                                action=action,
                                params=params,
                            )
                            for image in images
                        )
                    )
                    results = [result for result, _ in outcomes]
                    model_durations = [duration for _, duration in outcomes]
                except PayloadTooLargeError:
                    raise
                except ReloadAfterEvictionError as error:
                    failure = self._last_load_failure(route.registry_id)
                    if failure is None:
                        raise _not_ready_error() from error
                    raise _load_error(failure) from error
                except ValueError as error:
                    if isinstance(error.__cause__, ModelInputError):
                        raise error.__cause__ from error
                    raise ModelInputError(str(error)) from error
        except BaseException as error:
            if record:
                record_telemetry(
                    _record_model_invocation,
                    route,
                    images,
                    api_key,
                    time.perf_counter() - started,
                    None,
                    error,
                )
            else:
                _add_to_usage_call(images, None)
            raise
        duration = time.perf_counter() - started
        if record:
            record_telemetry(
                _record_model_invocation,
                route,
                images,
                api_key,
                duration,
                model_durations,
                None,
            )
        else:
            _add_to_usage_call(images, model_durations)
        if record:
            record_telemetry(
                telemetry.record_inference,
                requested_model_id_for(route.registry_id),
                duration,
            )

        return results, model_durations

    async def _infer_one(
        self,
        model_id: str,
        image: Any,
        *,
        action: str,
        params: dict,
    ) -> tuple[Any, Optional[float]]:
        infer_with_duration = getattr(self.gateway, "infer_with_duration", None)
        if infer_with_duration is None:
            result = await self.gateway.infer(
                model_id=model_id, image=image, action=action, params=params
            )

            return result, None

        outcome = await infer_with_duration(
            model_id=model_id, image=image, action=action, params=params
        )

        return outcome

    def usage_call(
        self,
        route: Route,
        api_key: Optional[str],
        *,
        images: Optional[list[Optional[ImagePayload]]] = None,
    ) -> Any:
        """Group the ``record=False`` calls made inside into one ``model`` row.

        Args:
            route: Route of the model the calls go to.
            api_key: Key the model is called with.
            images: Images of the whole call; see ``usage_call``.

        Returns:
            The context manager recording the combined call on exit.
        """
        return usage_call(route, api_key, images=images)

    async def infer_params_only(
        self,
        route: Route,
        api_key: Optional[str],
        action: str,
        params: dict,
        *,
        model_monitoring: bool = True,
        record: bool = True,
    ) -> Any:
        results = await self.infer(
            route,
            api_key,
            action,
            [None],
            params,
            model_monitoring=model_monitoring,
            record=record,
        )
        return results[0]

    async def fetch_image(self, url: str) -> bytes:
        if LEGACY_OFFLINE_MODE or not ALLOW_URL_INPUT:
            raise LegacyHTTPError(
                400, "Loading images from URLs is not allowed on this server."
            )
        images, error = await fetch_url_images([url])
        if error is not None:
            raise ImageFetchError(error.status_code, "Could not fetch image from URL.")
        return images[0]

    async def unload(self, model_id: str) -> None:
        route = self._routes.get(model_id)
        registry_id = (
            route.registry_id if route is not None else registry_id_for(model_id)
        )
        result = await self.gateway.unload(registry_id)
        self._record_unload(registry_id, result)
        for key in [
            key
            for key, cached in self._routes.items()
            if cached.registry_id == registry_id
        ]:
            del self._routes[key]
        self._loaded_ids.discard(registry_id)
        self._preloaded_ids.pop(registry_id, None)

    def _record_unload(self, registry_id: str, result: Any) -> None:
        if result and result[0] == "ok":
            record_telemetry(telemetry.record_model_unloaded, registry_id)

    async def remove(self, model_id: str) -> None:
        requested_id, instance = parse_registration_key(model_id)
        registry_id = routing_key(registry_id_for(requested_id), instance)
        if instance:
            legacy_model_id = model_id
        else:
            legacy_model_id = resolve_roboflow_model_alias(model_id)
        routes = {route.registry_id: route for route in await self.describe()}
        route = routes.get(registry_id)
        if route is None:
            return

        if route.requested_at:
            if legacy_model_id not in route.requested_at:
                return
            del route.requested_at[legacy_model_id]
            route.request_paths_by_id.pop(legacy_model_id, None)
            self._preloaded_ids.get(registry_id, {}).pop(legacy_model_id, None)
            if route.requested_at:
                return

        await self.unload(registry_id)

    async def flush_model_stream_pipeline(self, model_id: str) -> Optional[list]:
        flushed = await self.gateway.flush_model_stream_pipeline(model_id)

        return flushed

    async def shutdown_model_stream_pipeline(self, model_id: str) -> None:
        await self.gateway.shutdown_model_stream_pipeline(model_id)

        return None

    async def unload_all(self) -> None:
        models = await self._stats_models()
        for model_id in list(models):
            result = await self.gateway.unload(model_id)
            self._record_unload(model_id, result)
        self._routes.clear()
        self._loaded_ids.clear()
        self._preloaded_ids.clear()

    async def describe(self) -> list[Route]:
        models = await self._stats_models()
        routes: list[Route] = []
        for model_id, entry in models.items():
            cached = self._routes.get(model_id)
            if cached is not None and cached.registry_id == model_id:
                _apply_metadata(cached, entry)
                routes.append(cached)
                continue
            route = Route(
                model_id=model_id,
                registry_id=model_id,
                task_type="unknown",
                action="infer",
            )
            _apply_metadata(route, entry)
            route.task_type = _task_type_from_mro(route.model_mro_names)
            route.action = _DEFAULT_ACTION_BY_TASK_TYPE.get(route.task_type, "infer")
            routes.append(route)
        for route in routes:
            preloaded = self._preloaded_ids.get(route.registry_id, {})
            for preloaded_id, registered_at in preloaded.items():
                _record_latest(route.requested_at, preloaded_id, registered_at)
                alias = request_alias_for(preloaded_id)
                if alias is not None:
                    _record_latest(
                        route.request_aliases_by_id.setdefault(preloaded_id, {}),
                        alias,
                        registered_at,
                    )
            _drop_stale(route.requested_at, route.loaded_monotonic)
            for paths in route.request_paths_by_id.values():
                _drop_stale(paths, route.loaded_monotonic)
            for aliases in route.request_aliases_by_id.values():
                _drop_stale(aliases, route.loaded_monotonic)
        return routes

    def record_request(
        self,
        route: Route,
        model_id_as_requested: str,
        path: str,
        *,
        alias: Optional[str] = None,
        join_pending: bool = True,
    ) -> None:
        if model_id_as_requested:
            _remember_request(route, model_id_as_requested, path, alias)
            recorded_at = _clock()
            route.requested_at[model_id_as_requested] = recorded_at
            paths = route.request_paths_by_id.setdefault(model_id_as_requested, {})
            if path:
                paths[path] = recorded_at
            if alias is not None:
                route.request_aliases_by_id.setdefault(model_id_as_requested, {})[
                    alias
                ] = recorded_at
            if join_pending:
                self._join_pending_request(route, model_id_as_requested, recorded_at)

    def _hold_pending_request(
        self, row_key: Optional[str], path: str, alias: Optional[str]
    ) -> None:
        if not row_key:
            return

        paths, aliases = self._pending_requests.pop(row_key, (set(), set()))
        if path and len(paths) < _MAX_PENDING_VALUES_PER_KEY:
            paths.add(path)
        if alias is not None and len(aliases) < _MAX_PENDING_VALUES_PER_KEY:
            aliases.add(alias)
        self._pending_requests[row_key] = (paths, aliases)

        while len(self._pending_requests) > _MAX_PENDING_REQUEST_KEYS:
            del self._pending_requests[next(iter(self._pending_requests))]

    def _join_pending_request(
        self, route: Route, row_key: str, recorded_at: float
    ) -> None:
        if self._routes.get(route.registry_id) is not route:
            return

        pending_paths, pending_aliases = self._pending_requests.pop(
            row_key, (set(), set())
        )
        if pending_paths:
            paths = route.request_paths_by_id.setdefault(row_key, {})
            paths.update(dict.fromkeys(pending_paths, recorded_at))
        if pending_aliases:
            aliases = route.request_aliases_by_id.setdefault(row_key, {})
            aliases.update(dict.fromkeys(pending_aliases, recorded_at))

    def register_preloaded(self, model_id: str) -> None:
        self._preloaded_ids.setdefault(registry_id_for(model_id), {})[
            model_id
        ] = _clock()

    def _refresh_current_request(self, registry_id: str) -> None:
        current = _CURRENT_REQUEST.get() or {}
        for route, model_id_as_requested, path, alias in list(current.values()):
            if route.registry_id == registry_id:
                self.record_request(
                    route,
                    model_id_as_requested,
                    path,
                    alias=alias,
                    join_pending=False,
                )

    def __contains__(self, model_id: str) -> bool:
        route = self._routes.get(model_id)
        return route is not None and route.registry_id in self._loaded_ids

    async def _stat(
        self, model_id: str, registry_id: str, api_key: Optional[str]
    ) -> ModelStat:
        try:
            return await stat_model_details_while_checking_auth(
                CommonRequestParams(model_id=registry_id, api_key=api_key or "")
            )
        except LookupError:
            for family in (model_id.split("/")[0], registry_id.split("/")[0]):
                static = _CORE_MODEL_TASK_TYPES.get(family)
                if static is not None:
                    return ModelStat(
                        *static,
                        model_architecture=_CORE_MODEL_ARCHITECTURES.get(
                            family, family
                        ),
                        model_variant=registry_id.partition("/")[2] or None,
                    )
            raise

    async def _stats_models(self) -> dict:
        stats = await self.gateway.stats()
        models = stats.get("models") or {}
        self._loaded_ids = set(models)
        for route in self._routes.values():
            if route.registry_id not in self._loaded_ids:
                route.metadata_ts = 0.0
        return models

    def _metadata_expired(self, route: Route) -> bool:
        if route.metadata_ts == 0.0:
            return True
        return time.monotonic() - route.metadata_ts >= LEGACY_ROUTE_METADATA_TTL_S

    async def _refresh_metadata(self, route: Route) -> None:
        models = await self._stats_models()
        entry = models.get(route.registry_id)
        if entry is None:
            return
        _apply_metadata(route, entry)


async def _resolve_stub(
    model_id: str, registry_id: str, api_key: Optional[str]
) -> Route:
    stat = await _stat_stub(registry_id, api_key)
    if stat.task_type not in _STUB_TASK_TYPES:
        raise LegacyHTTPError(500, SERVICE_MISCONFIGURATION_MESSAGE)

    route = Route(
        model_id=model_id,
        registry_id=registry_id,
        task_type=stat.task_type,
        action=stat.default_action,
        is_stub=True,
    )
    _apply_stat(route, stat)

    return route


async def _stat_stub(registry_id: str, api_key: Optional[str]) -> ModelStat:
    if api_key is None:
        raise LegacyHTTPError(400, MISSING_API_KEY_MESSAGE)

    key = (registry_id, api_key)
    cached = _stub_stats.get(key)
    if cached is not None:
        return cached

    if LEGACY_OFFLINE_MODE:
        raise LegacyHTTPError(503, REGISTRY_UNREACHABLE_MESSAGE)

    dataset_id = registry_id.partition("/")[0]
    if not _PLATFORM_ID_PATTERN.fullmatch(dataset_id):
        raise LegacyHTTPError(404, NOT_FOUND_MESSAGE)

    task_type = await asyncio.to_thread(_fetch_project_task_type, dataset_id, api_key)
    stat = ModelStat(task_type, "infer", model_architecture=STUB_MODEL_ARCHITECTURE)
    _stub_stats.set(key, stat)

    return stat


def _fetch_project_task_type(dataset_id: str, api_key: str) -> str:
    workspace_id = _fetch_workspace_id(api_key)
    dataset_info = _get_platform_json(
        f"{API_BASE_URL}/{workspace_id}/{dataset_id}", api_key
    )
    project = dataset_info.get("project", {})
    if "type" not in project:
        logger.warning(
            "Project task type not defined for workspace=%s and dataset=%s, "
            "defaulting to %s.",
            workspace_id,
            dataset_id,
            _DEFAULT_PROJECT_TASK_TYPE,
        )
    task_type = project.get("type", _DEFAULT_PROJECT_TASK_TYPE)

    return task_type


def _fetch_workspace_id(api_key: str) -> str:
    if not api_key:
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE)

    workspace_id = _get_platform_json(f"{API_BASE_URL}/", api_key).get("workspace")
    if not isinstance(workspace_id, str) or not _PLATFORM_ID_PATTERN.fullmatch(
        workspace_id
    ):
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE)

    return workspace_id


def _get_platform_json(url: str, api_key: str) -> dict:
    full_url = platform_http.wrap_url(
        platform_http._add_params_to_url(
            url, [("api_key", api_key), ("nocache", "true")]
        )
    )
    response = platform_http._platform_request(
        "get",
        full_url,
        headers=platform_http.build_api_headers(),
        timeout=platform_http.API_REQUEST_TIMEOUT_S,
    )
    if response.status_code >= 400:
        raise _platform_error(response.status_code)
    try:
        payload = response.json()
    except ValueError as error:
        raise LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE) from error

    return payload


def _platform_error(status_code: int) -> LegacyHTTPError:
    if status_code == 401:
        return LegacyHTTPError(401, UNAUTHORIZED_MESSAGE)
    if status_code == 404:
        return LegacyHTTPError(404, NOT_FOUND_MESSAGE)
    if status_code in MODEL_ACCESS_ERROR_MESSAGES:
        return LegacyHTTPError(status_code, MODEL_ACCESS_ERROR_MESSAGES[status_code])

    return LegacyHTTPError(502, REGISTRY_REQUEST_FAILED_MESSAGE)


def _reset_stub_cache_for_tests() -> None:
    _stub_stats.clear()


def _not_ready_error() -> ModelNotReadyError:
    return ModelNotReadyError()


def _load_error(result: tuple) -> Exception:
    error = load_failure_error(result)
    if error is None:
        return LegacyHTTPError(500, MODEL_PACKAGE_BROKEN_MESSAGE)

    return error


def _apply_stat(route: Route, stat: Optional[ModelStat]) -> None:
    if stat is None:
        return
    route.task_type = stat.task_type
    route.action = stat.default_action or route.action
    route.model_architecture = stat.model_architecture
    route.model_variant = stat.model_variant


def _record_model_invocation(
    route: Route,
    images: list[Optional[ImagePayload]],
    api_key: Optional[str],
    duration: float,
    model_durations: Optional[list[Optional[float]]],
    error: Optional[BaseException],
) -> None:
    scope = USAGE_SCOPE.get()
    if scope is None:
        return

    details: dict[str, Any] = {}
    if route.model_architecture:
        details["model_architecture"] = route.model_architecture
    if route.model_variant:
        details["model_variant"] = route.model_variant
    details["task_type"] = route.task_type
    if route.input_height is not None and route.input_width is not None:
        details["model_input_height"] = route.input_height
        details["model_input_width"] = route.input_width
    present = [image for image in images if image is not None]
    frames = max(1, len(present))
    model_duration = execution_duration(
        _model_duration(duration, model_durations, error)
    )
    record_model_usage(
        scope,
        model_id=resolve_roboflow_model_alias(
            requested_model_id_for(route.registry_id)
        ),
        api_key=api_key,
        frames=frames,
        duration=model_duration,
        details=details,
        megapixel_buckets=_megapixel_buckets(
            route, images, frames, model_duration, model_durations
        ),
        error=error,
    )


def _add_to_usage_call(
    images: list[Optional[ImagePayload]],
    model_durations: Optional[list[Optional[float]]],
) -> None:
    call = _USAGE_CALL.get()
    if call is None:
        return

    call.add(images, model_durations)


def _model_duration(
    round_trip: float,
    model_durations: Optional[list[Optional[float]]],
    error: Optional[BaseException],
) -> float:
    if error is not None:
        return round_trip
    if not model_durations or any(value is None for value in model_durations):
        return round_trip

    total = float(sum(model_durations))

    return total


def _megapixel_buckets(
    route: Route,
    images: list[Optional[ImagePayload]],
    frames: int,
    duration: float,
    model_durations: Optional[list[Optional[float]]],
) -> dict[str, dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    present = [index for index, image in enumerate(images) if image is not None]
    if not present:
        add_megapixel_bucket(
            buckets, MEGAPIXEL_BUCKET_UNKNOWN, frames=frames, duration=duration
        )
        return buckets
    fixed = megapixel_bucket(route.input_height, route.input_width)
    if fixed != MEGAPIXEL_BUCKET_UNKNOWN:
        add_megapixel_bucket(buckets, fixed, frames=frames, duration=duration)
        return buckets

    weights = _duration_weights(present, model_durations)
    for index, weight in zip(present, weights):
        image = images[index]
        add_megapixel_bucket(
            buckets,
            megapixel_bucket(image.height, image.width),
            frames=1,
            duration=duration * weight,
        )

    return buckets


def _duration_weights(
    present: list[int], model_durations: Optional[list[Optional[float]]]
) -> list[float]:
    even = [1.0 / len(present)] * len(present)
    if not model_durations or len(model_durations) < len(present):
        return even
    measured = [model_durations[index] for index in present]
    if any(value is None for value in measured):
        return even
    total = float(sum(measured))
    if total <= 0:
        return even

    weights = [value / total for value in measured]

    return weights


def _apply_metadata(route: Route, entry: dict) -> None:
    route.actions = set(entry.get("actions") or {})
    route.class_names = entry.get("class_names")
    route.key_points_classes = entry.get("key_points_classes")
    route.video_sampling = entry.get("video_sampling")
    route.model_class_name = entry.get("model_class_name")
    route.model_mro_names = list(entry.get("model_mro_names") or [])
    route.class_colors = entry.get("class_colors")
    route.resolved_model = entry.get("resolved_model")
    route.input_height = entry.get("input_height")
    route.input_width = entry.get("input_width")
    route.vram_bytes = entry.get("vram_bytes")
    route.loaded_monotonic = entry.get("loaded_monotonic")
    route.stream_pipeline_depth = int(entry.get("stream_pipeline_depth") or 1)
    route.metadata_ts = time.monotonic()


def _clock() -> float:
    return time.monotonic()


def _record_latest(recorded: dict[str, float], key: str, recorded_at: float) -> None:
    recorded[key] = max(recorded_at, recorded.get(key, recorded_at))


def _drop_stale(recorded: dict[str, float], loaded_monotonic: Optional[float]) -> None:
    if loaded_monotonic is None:
        return

    for key in [
        key for key, recorded_at in recorded.items() if recorded_at < loaded_monotonic
    ]:
        del recorded[key]


def _task_type_from_mro(model_mro_names: list[str]) -> str:
    for name in model_mro_names:
        task_type = _TASK_TYPE_BY_MRO.get(name)
        if task_type is not None:
            return task_type
    return "unknown"


class SyncLegacyBridge:
    """Thread-side facade for Workflows; every method = LoopBridge.run(bridge.<method>(...), timeout)."""

    def __init__(self, bridge: LegacyModelBridge, loop_bridge: LoopBridge) -> None:
        self._bridge = bridge
        self._loop_bridge = loop_bridge
        self._usage_scope = USAGE_SCOPE.get()
        self.accepts_ndarray = bridge.accepts_ndarray

    def resolve(
        self, model_id, api_key, *, row_key=None, path="", alias=None, instance=""
    ) -> Route:
        set_requested_model_id(registry_id_for(model_id), requested_model_id=model_id)
        return self._run(
            self._bridge.resolve(
                model_id,
                api_key,
                row_key=row_key,
                path=path,
                alias=alias,
                instance=instance,
            )
        )

    def ensure_loaded(self, route, api_key) -> None:
        return self._run(self._bridge.ensure_loaded(route, api_key))

    def infer(self, route, api_key, action, images, params, *, record=True) -> list:
        return self._run(
            self._bridge.infer(route, api_key, action, images, params, record=record)
        )

    def infer_params_only(self, route, api_key, action, params, *, record=True) -> Any:
        return self._run(
            self._bridge.infer_params_only(
                route, api_key, action, params, record=record
            )
        )

    def fetch_image(self, url) -> bytes:
        return self._run(self._bridge.fetch_image(url))

    def flush_model_stream_pipeline(self, model_id) -> Optional[list]:
        return self._run(self._bridge.flush_model_stream_pipeline(model_id))

    def shutdown_model_stream_pipeline(self, model_id) -> None:
        return self._run(self._bridge.shutdown_model_stream_pipeline(model_id))

    def record_request(self, route, model_id_as_requested, path, *, alias=None) -> None:
        if not model_id_as_requested:
            return

        _remember_request(route, model_id_as_requested, path, alias)
        self._run(self._record_request(route, model_id_as_requested, path, alias))

    async def _record_request(self, route, model_id_as_requested, path, alias) -> None:
        self._bridge.record_request(route, model_id_as_requested, path, alias=alias)

    @contextmanager
    def usage_call(self, route, api_key, *, images=None) -> Iterator[None]:
        with bound_scope(self._usage_scope):
            with usage_call(route, api_key, images=images):
                yield

    def __contains__(self, model_id) -> bool:
        return model_id in self._bridge

    def _run(self, coro) -> Any:
        return self._loop_bridge.run(
            self._with_scope(coro, _USAGE_CALL.get()), _sync_timeout()
        )

    async def _with_scope(self, coro, call) -> Any:
        token = _USAGE_CALL.set(call)
        try:
            with bound_scope(self._usage_scope):
                return await coro
        finally:
            _USAGE_CALL.reset(token)


def _sync_timeout() -> float:
    return LEGACY_LOAD_TIMEOUT_S + INFER_TIMEOUT_S + _SYNC_TIMEOUT_MARGIN_S
