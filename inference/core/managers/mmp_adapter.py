from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from inference_server.proxies.mmp_client import MMPClient

from inference.core.exceptions import (
    InferenceModelNotFound,
    ModelDeploymentNotSupportedError,
)
from inference.core.logger import logger
from inference.core.managers import mmp_translation as translation
from inference.core.managers.entities import ModelDescription
from inference.core.registries.roboflow import ModelEndpointType

_SYNC_BRIDGE_EXTRA_TIMEOUT_S = 30.0


class _InertModelStub:
    """Placeholder for a model served by the MMP; no legacy model attributes.

    Deliberately lacks ``flush`` so consumers probing for stream-pipelining
    support treat the model as not pipelined.
    """

    __slots__ = ("model_id",)

    def __init__(self, model_id: str):
        self.model_id = model_id


def _set_if_field(response: Any, name: str, value: Any) -> None:
    fields = getattr(type(response), "model_fields", None)
    if fields is None or name in fields:
        setattr(response, name, value)


def _unsupported(model_id: str) -> ModelDeploymentNotSupportedError:
    return ModelDeploymentNotSupportedError(
        f"Model '{model_id}' cannot be served by the MMP inference backend"
    )


class ModelManagerAdapter:
    """Routes legacy model-manager calls to the ModelManagerProcess.

    Holds the fully-decorated legacy stack only for non-inference delegation
    (pingback, request metadata); the legacy stack loads no models while this
    adapter is in front of it.
    """

    def __init__(self, legacy_stack, mmp_client: Optional["MMPClient"] = None):
        self._legacy = legacy_stack
        if mmp_client is None:
            from inference_server.proxies.mmp_client import MMPClient

            mmp_client = MMPClient()
        self._client = mmp_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._routes: Dict[str, dict] = {}

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._client.start()

    async def shutdown(self) -> None:
        await self._client.shutdown()

    # ------------------------------------------------------------------
    # sync -> async bridge
    # ------------------------------------------------------------------

    def _run_sync(self, coro):
        if self._loop is None:
            coro.close()
            raise RuntimeError("ModelManagerAdapter used before start()")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            coro.close()
            raise RuntimeError(
                "ModelManagerAdapter sync bridge called on its own event loop; "
                "this would deadlock"
            )
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        timeout = (
            self._client.load_wait_s
            + self._client.infer_timeout_s
            + _SYNC_BRIDGE_EXTRA_TIMEOUT_S
        )
        return future.result(timeout=timeout)

    # ------------------------------------------------------------------
    # routing
    # ------------------------------------------------------------------

    async def _resolve_route_async(
        self, model_id: str, api_key: Optional[str]
    ) -> dict:
        route = self._routes.get(model_id)
        if route is not None:
            if not route["supported"]:
                raise _unsupported(model_id)
            return route
        if model_id == "passthrough" or model_id.startswith("passthrough/"):
            self._routes[model_id] = {"supported": False}
            raise _unsupported(model_id)
        task_type, default_action = await translation.stat_model(
            model_id=model_id, api_key=api_key or ""
        )
        terminal = {
            "supported": False,
            "task_type": task_type,
            "action": default_action,
        }
        if task_type not in translation.IMPLEMENTED_TASK_TYPES:
            self._routes[model_id] = terminal
            raise _unsupported(model_id)
        result = await self._client.load(model_id, api_key or "")
        translation.raise_for_lifecycle_result(result, model_id)
        interface = await self._client.interface(model_id)
        tasks = set(interface.get("tasks", {}))
        if not tasks.intersection(
            translation.implemented_actions(task_type)
        ):
            await self._client.unload(model_id)
            self._routes[model_id] = terminal
            raise _unsupported(model_id)
        stats = await self._client.stats()
        model_entry = stats.get("mmp_models", {}).get(model_id, {})
        key_points_classes = model_entry.get("key_points_classes")
        if task_type == "keypoint-detection" and key_points_classes is None:
            await self._client.unload(model_id)
            self._routes[model_id] = terminal
            raise _unsupported(model_id)
        if (
            task_type == "vlm"
            and model_entry.get("model_class_name")
            in translation.VLM_UNSUPPORTED_MODEL_CLASSES
        ):
            await self._client.unload(model_id)
            self._routes[model_id] = terminal
            raise _unsupported(model_id)
        route = {
            "supported": True,
            "mmp_model_id": model_id,
            "task_type": task_type,
            "action": default_action,
            "tasks": tasks,
            "class_names": model_entry.get("class_names"),
            "key_points_classes": key_points_classes,
        }
        self._routes[model_id] = route
        return route

    async def _infer_new_path(self, model_id: str, request, **kwargs):
        route = await self._resolve_route_async(
            model_id, getattr(request, "api_key", None)
        )
        action = translation.resolve_request_action(route, request)
        if (route["task_type"], action) not in translation.IMPLEMENTED_ROUTES or (
            action not in route["tasks"]
        ):
            raise _unsupported(model_id)
        translation.ensure_request_supported(model_id, request)
        is_batch = isinstance(request.image, list)
        images = request.image if is_batch else [request.image]
        forwarded = [translation.forward_image(image) for image in images]
        params = translation.build_task_params(
            route["task_type"], action, request
        )
        t_start = time.perf_counter()
        ensure = await self._client.ensure_loaded(
            route["mmp_model_id"], api_key=getattr(request, "api_key", None) or ""
        )
        translation.raise_for_lifecycle_result(ensure, model_id)
        predictions = await self._fan_out_infer(
            model_id, route, action, forwarded, params
        )
        elapsed = time.perf_counter() - t_start
        responses = []
        for prediction, (_, dims) in zip(predictions, forwarded):
            response = translation.repack_prediction(
                route["task_type"], action, prediction, dims, route, request
            )
            _set_if_field(response, "time", elapsed)
            _set_if_field(response, "inference_id", request.id)
            responses.append(response)
        return responses if is_batch else responses[0]

    async def _fan_out_infer(
        self, model_id: str, route: dict, action: str, forwarded: list, params: dict
    ) -> list:
        concurrency = min(
            len(forwarded), max(1, getattr(self._client, "n_slots", 1) or 1)
        )
        semaphore = asyncio.Semaphore(concurrency)

        async def infer_one(image_bytes: bytes):
            async with semaphore:
                return await self._client.infer(
                    model_id=route["mmp_model_id"],
                    image=image_bytes,
                    task=action,
                    params=params,
                )

        results = await asyncio.gather(
            *(infer_one(image_bytes) for image_bytes, _ in forwarded),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                translated = translation.translate_infer_error(result, model_id)
                if translated is result:
                    raise result
                raise translated from result
        return list(results)

    # ------------------------------------------------------------------
    # model operations
    # ------------------------------------------------------------------

    def add_model(
        self,
        model_id: str,
        api_key: str,
        model_id_alias: Optional[str] = None,
        endpoint_type: ModelEndpointType = ModelEndpointType.ORT,
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> None:
        route = self._run_sync(self._resolve_route_async(model_id, api_key))
        if model_id_alias is not None:
            self._routes[model_id_alias] = route

    async def infer_from_request(self, model_id: str, request, **kwargs):
        return await self._infer_new_path(model_id, request, **kwargs)

    def infer_from_request_sync(self, model_id: str, request, **kwargs):
        return self._run_sync(self._infer_new_path(model_id, request, **kwargs))

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        route = self._routes.get(model_id)
        if route is None:
            route = self._run_sync(self._resolve_route_async(model_id, api_key))
        if not route.get("supported"):
            raise _unsupported(model_id)
        return route["task_type"]

    def get_class_names(self, model_id: str):
        route = self._routes.get(model_id)
        if route is None or not route.get("supported"):
            raise _unsupported(model_id)
        return route["class_names"]

    def describe_models(self) -> List[ModelDescription]:
        return [
            ModelDescription(
                model_id=model_id,
                task_type=route["task_type"],
                batch_size=None,
                input_height=None,
                input_width=None,
            )
            for model_id, route in self._routes.items()
            if route.get("supported")
        ]

    def remove(self, model_id: str, delete_from_disk: bool = True) -> None:
        route = self._routes.get(model_id)
        if route is None or not route.get("supported"):
            logger.warning(
                f"Attempted to remove model with id {model_id}, but it is not loaded. Skipping..."
            )
            return
        self._run_sync(self._client.unload(route["mmp_model_id"]))
        for key in [k for k, r in self._routes.items() if r is route]:
            self._routes.pop(key, None)

    def clear(self) -> None:
        for model_id in list(self._routes):
            if self._routes.get(model_id, {}).get("supported"):
                self.remove(model_id)
        self._routes.clear()

    def pin_model(self, model_id: str) -> None:
        pass

    # ------------------------------------------------------------------
    # legacy-only operations, never reached by scoped HTTP consumers
    # ------------------------------------------------------------------

    def predict(self, model_id: str, *args, **kwargs):
        raise _unsupported(model_id)

    def model_infer_sync(self, model_id: str, request, **kwargs):
        raise _unsupported(model_id)

    def preprocess(self, model_id: str, request):
        raise _unsupported(model_id)

    def postprocess(self, model_id: str, *args, **kwargs):
        raise _unsupported(model_id)

    def make_response(self, model_id: str, *args, **kwargs):
        raise _unsupported(model_id)

    # ------------------------------------------------------------------
    # non-infer delegation + container protocol
    # ------------------------------------------------------------------

    def init_pingback(self):
        self._legacy.init_pingback()

    def record_request_metadata(self, *args, **kwargs) -> None:
        self._legacy.record_request_metadata(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._legacy, name)

    def _supported_ids(self) -> List[str]:
        return [
            model_id
            for model_id, route in self._routes.items()
            if route.get("supported")
        ]

    def __contains__(self, model_id: str) -> bool:
        route = self._routes.get(model_id)
        return bool(route and route.get("supported"))

    def __getitem__(self, key: str) -> _InertModelStub:
        if key not in self:
            raise InferenceModelNotFound(f"Model with id {key} not loaded.")
        return _InertModelStub(key)

    def __len__(self) -> int:
        return len(self._supported_ids())

    def keys(self):
        return self._supported_ids()

    def models(self) -> Dict[str, _InertModelStub]:
        return {
            model_id: _InertModelStub(model_id) for model_id in self._supported_ids()
        }
