from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from inference_sdk.http.utils.aliases import resolve_roboflow_model_alias

from inference_server.configuration import (
    ALLOW_URL_INPUT,
    INFER_TIMEOUT_S,
    LEGACY_LOAD_POLL_INTERVAL_S,
    LEGACY_LOAD_TIMEOUT_S,
    LEGACY_ROUTE_METADATA_TTL_S,
    OFFLINE_MODE,
)
from inference_server.framework.entities import CommonRequestParams
from inference_server.framework.fanout import gather_bounded
from inference_server.framework.input_parsers.url_fetch import fetch_images_from_urls
from inference_server.framework.model_stat import stat_model_while_checking_auth
from inference_server.gateway import ModelManagerGateway
from inference_server.legacy.common import ImagePayload
from inference_server.legacy.entities import ResolvedModel
from inference_server.legacy.errors import (
    MODEL_PACKAGE_BROKEN_MESSAGE,
    LegacyHTTPError,
)

logger = logging.getLogger(__name__)

_ERR_NOT_LOADED = 6
_SYNC_TIMEOUT_MARGIN_S = 30


@dataclass
class Route:
    model_id: str
    registry_id: str
    task_type: str
    action: str
    actions: set[str] = field(default_factory=set)
    class_names: Optional[list[str]] = None
    key_points_classes: Optional[list[list[str]]] = None
    model_class_name: Optional[str] = None
    model_mro_names: list[str] = field(default_factory=list)
    class_colors: Optional[dict] = None
    request_aliases: set[str] = field(default_factory=set)
    request_paths: set[str] = field(default_factory=set)
    resolved_model: Optional[dict] = None
    metadata_ts: float = 0.0


_CORE_MODEL_TASK_TYPES: dict[str, tuple[str, str]] = {
    "clip": ("embedding", "embed_images"),
    "perception_encoder": ("embedding", "embed_images"),
    "sam": ("interactive-instance-segmentation", "embed"),
    "sam2": ("interactive-instance-segmentation", "embed"),
    "sam3": ("interactive-instance-segmentation", "embed"),
    "doctr": ("structured-ocr", "infer"),
    "easy_ocr": ("structured-ocr", "infer"),
    "trocr": ("text-only-ocr", "infer"),
    "yolo_world": ("open-vocabulary-object-detection", "infer"),
    "grounding_dino": ("open-vocabulary-object-detection", "infer"),
    "depth-anything-v2": ("depth-estimation", "infer"),
    "depth-anything-v3": ("depth-estimation", "infer"),
    "moondream2": ("vlm", "prompt"),
    "smolvlm2": ("vlm", "prompt"),
}

_REGISTRY_ID_ALIASES = {"perception_encoder": "perception-encoder"}

_TASK_TYPE_BY_MRO = {
    "ObjectDetectionModel": "object-detection",
    "OpenVocabularyObjectDetectionModel": "open-vocabulary-object-detection",
    "InstanceSegmentationModel": "instance-segmentation",
    "KeyPointsDetectionModel": "keypoint-detection",
    "ClassificationModel": "classification",
    "MultiLabelClassificationModel": "multi-label-classification",
    "SemanticSegmentationModel": "semantic-segmentation",
    "DepthEstimationModel": "depth-estimation",
    "TextImageEmbeddingModel": "embedding",
    "StructuredOCRModel": "structured-ocr",
    "TextOnlyOCRModel": "text-only-ocr",
    "SAMTorch": "interactive-instance-segmentation",
    "SAM2Torch": "interactive-instance-segmentation",
    "SAM3Torch": "interactive-instance-segmentation",
}

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

    async def resolve(self, model_id: str, api_key: Optional[str]) -> Route:
        registry_id = registry_id_for(model_id)
        task_type: Optional[str] = None
        action: Optional[str] = None
        if not OFFLINE_MODE:
            task_type, action = await self._stat(model_id, registry_id, api_key)
        route = self._routes.get(registry_id)
        if route is None:
            route = Route(
                model_id=model_id,
                registry_id=registry_id,
                task_type=task_type or "unknown",
                action=action or "infer",
            )
        else:
            _apply_stat(route, task_type, action)
        if OFFLINE_MODE:
            try:
                await self.ensure_loaded(route, api_key)
            except LegacyHTTPError as error:
                raise LegacyHTTPError(
                    404, f"Model {model_id} not available offline"
                ) from error
        else:
            await self.ensure_loaded(route, api_key)
        route = self._adopt_canonical(route, task_type, action)
        if self._metadata_expired(route) or route.registry_id not in self._loaded_ids:
            await self._refresh_metadata(route)
        route = self._adopt_canonical(route, task_type, action)
        if OFFLINE_MODE and task_type is None:
            route.task_type = _task_type_from_mro(route.model_mro_names)
            route.action = _DEFAULT_ACTION_BY_TASK_TYPE.get(route.task_type, "infer")
        self._routes[registry_id] = route
        self._routes[model_id] = route
        return route

    def _adopt_canonical(
        self, route: Route, task_type: Optional[str], action: Optional[str]
    ) -> Route:
        canonical = self._routes.get(route.registry_id)
        if canonical is None or canonical is route:
            return route
        _apply_stat(canonical, task_type, action)
        return canonical

    async def ensure_loaded(self, route: Route, api_key: Optional[str]) -> None:
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
                    raise LegacyHTTPError(
                        503, "Model is temporarily not ready - retry request."
                    )
                raise LegacyHTTPError(500, MODEL_PACKAGE_BROKEN_MESSAGE)
            if time.monotonic() >= deadline:
                raise LegacyHTTPError(
                    503, "Model is temporarily not ready - retry request."
                )
            await asyncio.sleep(LEGACY_LOAD_POLL_INTERVAL_S)

    async def load_pinned(self, model_id: str, api_key: Optional[str]) -> Route:
        route = await self.resolve(model_id, api_key)
        result = await self.gateway.load(
            route.registry_id, api_key or "", timeout_s=LEGACY_LOAD_TIMEOUT_S
        )
        state = result[0] if result else "error"
        if state != "ok":
            raise LegacyHTTPError(500, MODEL_PACKAGE_BROKEN_MESSAGE)
        return route

    async def infer(
        self,
        route: Route,
        api_key: Optional[str],
        action: str,
        images: list[Optional[ImagePayload]],
        params: dict,
    ) -> list[Any]:
        await self.ensure_loaded(route, api_key)
        return await gather_bounded(
            *(
                self.gateway.infer(
                    model_id=route.registry_id,
                    image=image.data if image is not None else None,
                    action=action,
                    params=params,
                )
                for image in images
            )
        )

    async def infer_params_only(
        self, route: Route, api_key: Optional[str], action: str, params: dict
    ) -> Any:
        results = await self.infer(route, api_key, action, [None], params)
        return results[0]

    async def fetch_image(self, url: str) -> bytes:
        if OFFLINE_MODE or not ALLOW_URL_INPUT:
            raise LegacyHTTPError(
                400, "Loading images from URLs is not allowed on this server."
            )
        images, error = await fetch_images_from_urls([url])
        if error is not None:
            raise LegacyHTTPError(error.status_code, "Could not fetch image from URL.")
        return images[0]

    async def unload(self, model_id: str) -> None:
        route = self._routes.get(model_id)
        registry_id = (
            route.registry_id if route is not None else registry_id_for(model_id)
        )
        await self.gateway.unload(registry_id)
        for key in [
            key
            for key, cached in self._routes.items()
            if cached.registry_id == registry_id
        ]:
            del self._routes[key]
        self._loaded_ids.discard(registry_id)

    async def unload_all(self) -> None:
        models = await self._stats_models()
        for model_id in list(models):
            await self.gateway.unload(model_id)
        self._routes.clear()
        self._loaded_ids.clear()

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
        return routes

    def record_request(
        self, route: Route, model_id_as_requested: str, path: str
    ) -> None:
        if model_id_as_requested and model_id_as_requested != route.registry_id:
            route.request_aliases.add(model_id_as_requested)
        if path:
            route.request_paths.add(path)

    def __contains__(self, model_id: str) -> bool:
        route = self._routes.get(model_id)
        return route is not None and route.registry_id in self._loaded_ids

    async def _stat(
        self, model_id: str, registry_id: str, api_key: Optional[str]
    ) -> tuple[str, str]:
        try:
            return await stat_model_while_checking_auth(
                CommonRequestParams(model_id=registry_id, api_key=api_key or "")
            )
        except LookupError:
            static = _CORE_MODEL_TASK_TYPES.get(
                model_id.split("/")[0]
            ) or _CORE_MODEL_TASK_TYPES.get(registry_id.split("/")[0])
            if static is None:
                raise
            return static

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


def _apply_stat(route: Route, task_type: Optional[str], action: Optional[str]) -> None:
    if task_type is None:
        return
    route.task_type = task_type
    route.action = action or route.action


def _apply_metadata(route: Route, entry: dict) -> None:
    route.actions = set(entry.get("actions") or {})
    route.class_names = entry.get("class_names")
    route.key_points_classes = entry.get("key_points_classes")
    route.model_class_name = entry.get("model_class_name")
    route.model_mro_names = list(entry.get("model_mro_names") or [])
    route.class_colors = entry.get("class_colors")
    route.resolved_model = entry.get("resolved_model")
    route.metadata_ts = time.monotonic()


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
        self.accepts_ndarray = bridge.accepts_ndarray

    def resolve(self, model_id, api_key) -> Route:
        return self._run(self._bridge.resolve(model_id, api_key))

    def ensure_loaded(self, route, api_key) -> None:
        return self._run(self._bridge.ensure_loaded(route, api_key))

    def infer(self, route, api_key, action, images, params) -> list:
        return self._run(self._bridge.infer(route, api_key, action, images, params))

    def infer_params_only(self, route, api_key, action, params) -> Any:
        return self._run(self._bridge.infer_params_only(route, api_key, action, params))

    def fetch_image(self, url) -> bytes:
        return self._run(self._bridge.fetch_image(url))

    def __contains__(self, model_id) -> bool:
        return model_id in self._bridge

    def _run(self, coro) -> Any:
        return self._loop_bridge.run(coro, _sync_timeout())


def _sync_timeout() -> float:
    return LEGACY_LOAD_TIMEOUT_S + INFER_TIMEOUT_S + _SYNC_TIMEOUT_MARGIN_S
