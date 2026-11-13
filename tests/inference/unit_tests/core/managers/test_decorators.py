from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from inference.core.managers.base import ModelManager
from inference.core.managers.decorators import fixed_size_cache
from inference.core.managers.decorators.base import ModelManagerDecorator
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
from inference.core.managers.decorators.locked_load import (
    LockedLoadModelManagerDecorator,
)
from inference.core.managers.model_load_collector import (
    RequestModelIds,
    current_request_path,
    request_model_ids,
)
from inference.core.models.embeddings import model_cache_key


def test_model_manager_decorator_records_request_metadata_for_warm_model() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"some/1": MagicMock()}
    decorator = ModelManagerDecorator(model_manager)
    request_path_token = current_request_path.set("/sam3/concept_segment")
    ids = RequestModelIds()
    ids_token = request_model_ids.set(ids)

    try:
        decorator.add_model(
            model_id="some/1",
            api_key="key",
            model_id_alias="sam3/sam3_final",
        )
    finally:
        request_model_ids.reset(ids_token)
        current_request_path.reset(request_path_token)

    [description] = model_manager.describe_models()
    assert description.model_id == "some/1"
    assert description.request_aliases == ["sam3/sam3_final"]
    assert description.request_paths == ["/sam3/concept_segment"]
    assert ids.get_ids() == {"some/1"}


def test_model_manager_decorator_exposes_owned_artifact_cache() -> None:
    artifact_cache = MagicMock()
    model_manager = ModelManager(
        model_registry=MagicMock(),
        content_addressed_artifact_cache=artifact_cache,
    )

    decorator = ModelManagerDecorator(model_manager)

    assert decorator.content_addressed_artifact_cache is artifact_cache


def test_fixed_size_cache_records_request_metadata_for_warm_model() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"sam3/sam3_interactive": MagicMock()}
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    token = current_request_path.set("/sam3/embed_image")

    try:
        decorator.add_model(
            model_id="sam3/sam3_final",
            api_key="key",
            model_id_alias="sam3/sam3_interactive",
        )
        decorator.add_model(
            model_id="sam3/sam3_final",
            api_key="key",
            model_id_alias="sam3/sam3_interactive",
        )
    finally:
        current_request_path.reset(token)

    [description] = model_manager.describe_models()
    assert description.model_id == "sam3/sam3_interactive"
    assert description.request_aliases == ["sam3/sam3_final"]
    assert description.request_paths == ["/sam3/embed_image"]


def test_fixed_size_cache_skips_online_authorization_in_offline_mode() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    decorator = WithFixedSizeCache(model_manager, max_size=8)

    with patch.object(
        fixed_size_cache,
        "MODELS_CACHE_AUTH_ENABLED",
        True,
    ), patch.object(
        fixed_size_cache,
        "OFFLINE_MODE",
        True,
    ), patch.object(
        fixed_size_cache,
        "_check_if_api_key_has_access_to_model",
    ) as access_check_mock:
        decorator.add_model(model_id="some/1", api_key="key")

    access_check_mock.assert_not_called()
    assert "some/1" in model_manager.models()


def test_nested_decorators_refresh_native_embedding_model_cache() -> None:
    embedding_key = model_cache_key("some/1", ["image_embeddings"], "logits")
    model = MagicMock()
    result = {"embeddings": object(), "embedding_info": {"output_type": "logits"}}
    model.run_tensor_native_embeddings.return_value = result
    base_manager = ModelManager(model_registry=MagicMock())
    base_manager._models = {embedding_key: model, "other/1": MagicMock()}
    decorator = WithFixedSizeCache(
        LockedLoadModelManagerDecorator(base_manager), max_size=2
    )
    images = [object()]

    actual = decorator.run_tensor_native_embeddings(
        model_id=embedding_key,
        images=images,
        output_type="logits",
        input_color_format="rgb",
    )

    assert actual is result
    assert list(decorator._key_queue) == ["other/1", embedding_key]
    model.run_tensor_native_embeddings.assert_called_once_with(
        images=images, output_type="logits", input_color_format="rgb"
    )


def test_nested_decorators_record_request_metadata_for_warm_model() -> None:
    base_manager = ModelManager(model_registry=MagicMock())
    base_manager._models = {"some/1": MagicMock()}
    decorator = WithFixedSizeCache(
        LockedLoadModelManagerDecorator(base_manager), max_size=8
    )
    path_token = current_request_path.set("/infer/object_detection")
    ids = RequestModelIds()
    ids_token = request_model_ids.set(ids)

    try:
        decorator.add_model(model_id="some/1", api_key="key")
    finally:
        request_model_ids.reset(ids_token)
        current_request_path.reset(path_token)

    [description] = base_manager.describe_models()
    assert description.model_id == "some/1"
    assert description.request_aliases == []
    assert description.request_paths == ["/infer/object_detection"]
    assert ids.get_ids() == {"some/1"}


@pytest.mark.parametrize("disk_cache_cleanup", [True, False])
def test_fixed_size_cache_clear_passes_disk_cache_cleanup_to_models(
    disk_cache_cleanup: bool,
) -> None:
    # given
    model_manager = ModelManager(model_registry=MagicMock())
    model = MagicMock()
    model_manager._models = {"some/1": model}
    decorator = WithFixedSizeCache(model_manager, max_size=8)

    # when
    with patch.object(fixed_size_cache, "DISK_CACHE_CLEANUP", disk_cache_cleanup):
        decorator.clear()

    # then
    model.clear_cache.assert_called_once_with(delete_from_disk=disk_cache_cleanup)
    assert model_manager.models() == {}


def test_fixed_size_cache_evicts_unpinned_models_when_adding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {
        "pinned/1": MagicMock(),
        "other/1": MagicMock(),
        "other/2": MagicMock(),
        "other/3": MagicMock(),
        "other/4": MagicMock(),
    }
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    decorator.pin_model("pinned/1")
    monkeypatch.setattr(fixed_size_cache, "MEMORY_FREE_THRESHOLD", 0.3)
    monkeypatch.setattr(
        decorator, "memory_pressure_detected", MagicMock(side_effect=[True, False])
    )

    # when
    decorator.add_model(model_id="new/1", api_key="key")

    # then
    assert list(model_manager.models()) == ["pinned/1", "other/4", "new/1"]
    assert list(decorator._key_queue) == ["pinned/1", "other/4", "new/1"]


def test_fixed_size_cache_evicts_models_and_retries_after_cuda_out_of_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    model = MagicMock()
    model.infer_from_request.side_effect = [
        torch.cuda.OutOfMemoryError("CUDA out of memory."),
        "response",
    ]
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {
        "pinned/1": MagicMock(),
        "other/1": MagicMock(),
        "other/2": MagicMock(),
        "other/3": MagicMock(),
        "other/4": MagicMock(),
        "sam3/sam3_interactive": model,
    }
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    decorator.pin_model("pinned/1")
    request = SimpleNamespace(api_key="key", disable_model_monitoring=True)
    monkeypatch.setattr(fixed_size_cache, "MEMORY_FREE_THRESHOLD", 0.3)
    monkeypatch.setattr(
        decorator, "memory_pressure_detected", MagicMock(side_effect=[True, False])
    )

    # when
    response = decorator.infer_from_request_sync("sam3/sam3_interactive", request)

    # then
    assert response == "response"
    assert model.infer_from_request.call_count == 2
    assert list(model_manager.models()) == [
        "pinned/1",
        "other/4",
        "sam3/sam3_interactive",
    ]
    assert list(decorator._key_queue) == [
        "pinned/1",
        "other/4",
        "sam3/sam3_interactive",
    ]


def test_fixed_size_cache_raises_cuda_oom_when_no_model_can_be_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # given
    model = MagicMock()
    model.infer_from_request.side_effect = torch.cuda.OutOfMemoryError(
        "CUDA out of memory."
    )
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"pinned/1": MagicMock(), "some/1": model}
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    decorator.pin_model("pinned/1")
    request = SimpleNamespace(api_key="key", disable_model_monitoring=True)
    monkeypatch.setattr(fixed_size_cache, "MEMORY_FREE_THRESHOLD", 0.3)
    monkeypatch.setattr(
        decorator, "memory_pressure_detected", MagicMock(return_value=True)
    )

    # when
    with pytest.raises(torch.cuda.OutOfMemoryError):
        decorator.infer_from_request_sync("some/1", request)

    # then
    assert model.infer_from_request.call_count == 1
    assert list(model_manager.models()) == ["pinned/1", "some/1"]
    assert list(decorator._key_queue) == ["pinned/1", "some/1"]


@pytest.mark.parametrize(
    "memory_free_threshold, error",
    [
        (0.0, torch.cuda.OutOfMemoryError("CUDA out of memory.")),
        (0.3, ValueError("Invalid prompt.")),
    ],
    ids=["threshold_unset", "not_out_of_memory"],
)
def test_fixed_size_cache_does_not_evict_models_after_other_inference_errors(
    monkeypatch: pytest.MonkeyPatch,
    memory_free_threshold: float,
    error: Exception,
) -> None:
    # given
    model = MagicMock()
    model.infer_from_request.side_effect = error
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"other/1": MagicMock(), "some/1": model}
    decorator = WithFixedSizeCache(model_manager, max_size=8)
    request = SimpleNamespace(api_key="key", disable_model_monitoring=True)
    monkeypatch.setattr(
        fixed_size_cache, "MEMORY_FREE_THRESHOLD", memory_free_threshold
    )
    monkeypatch.setattr(
        decorator, "memory_pressure_detected", MagicMock(return_value=True)
    )

    # when
    with pytest.raises(type(error)):
        decorator.infer_from_request_sync("some/1", request)

    # then
    assert model.infer_from_request.call_count == 1
    assert list(model_manager.models()) == ["other/1", "some/1"]
