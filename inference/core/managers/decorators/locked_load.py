from inference.core.cache.shared_lock import SharedLock
from inference.core.entities.requests.inference import InferenceRequest
from inference.core.entities.responses.inference import InferenceResponse
from inference.core.managers.decorators.base import ModelManagerDecorator

add_rem_model_lock_str = lambda z: f"locks:model-add-rem:{z}"
infer_share_key = lambda z: f"locks:model-infer:{z}"


class LockedLoadModelManagerDecorator(ModelManagerDecorator):
    """Must acquire lock to load model"""

    def add_model(self, model_id: str, api_key: str, model_id_alias=None):
        queue_id = self._resolve_queue_id(
            model_id=model_id, model_id_alias=model_id_alias
        )
        with SharedLock(add_rem_model_lock_str(queue_id), expire=180.0):
            return super().add_model(model_id, api_key, model_id_alias=model_id_alias)

    def remove(self, model_id: str):
        with SharedLock(add_rem_model_lock_str(model_id), expire=180.0):
            return super().remove(model_id)

    async def infer_from_request(
        self, model_id: str, request: InferenceRequest, **kwargs
    ) -> InferenceResponse:
        with SharedLock(
            add_rem_model_lock_str(model_id),
            share_key=infer_share_key(model_id),
            expire=180.0,
        ):
            return await super().infer_from_request(model_id, request, **kwargs)
