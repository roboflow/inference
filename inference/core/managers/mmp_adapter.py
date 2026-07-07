from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from inference_server.proxies.mmp_client import MMPClient

from inference.core.exceptions import (
    InferenceModelNotFound,
    ModelDeploymentNotSupportedError,
)
from inference.core.logger import logger
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
        raise _unsupported(model_id)

    async def infer_from_request(self, model_id: str, request, **kwargs):
        raise _unsupported(model_id)

    def infer_from_request_sync(self, model_id: str, request, **kwargs):
        raise _unsupported(model_id)

    def get_task_type(self, model_id: str, api_key: str = None) -> str:
        raise _unsupported(model_id)

    def get_class_names(self, model_id: str):
        raise _unsupported(model_id)

    def describe_models(self) -> List[ModelDescription]:
        return []

    def remove(self, model_id: str, delete_from_disk: bool = True) -> None:
        if model_id not in self._routes:
            logger.warning(
                f"Attempted to remove model with id {model_id}, but it is not loaded. Skipping..."
            )
            return
        self._run_sync(self._client.unload(model_id))
        self._routes.pop(model_id, None)

    def clear(self) -> None:
        for model_id in list(self._routes):
            self.remove(model_id)

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

    def __contains__(self, model_id: str) -> bool:
        return model_id in self._routes

    def __getitem__(self, key: str) -> _InertModelStub:
        if key not in self._routes:
            raise InferenceModelNotFound(f"Model with id {key} not loaded.")
        return _InertModelStub(key)

    def __len__(self) -> int:
        return len(self._routes)

    def keys(self):
        return self._routes.keys()

    def models(self) -> Dict[str, _InertModelStub]:
        return {model_id: _InertModelStub(model_id) for model_id in self._routes}
