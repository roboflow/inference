from queue import Queue
from threading import Event, Lock, Thread, current_thread
from typing import Callable, List
from unittest.mock import MagicMock, patch

import pytest

from inference.core.managers import base as base_module
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

MODEL_ID = "some/1"
ALIAS_ID = "alias/1"
THREAD_TIMEOUT = 2.0
MUTATION_ORDER_OBSERVATION_TIMEOUT = 0.2


class _ObservableLock:
    def __init__(
        self, acquisition_attempts: Queue, observed_thread_name: str = "add-model"
    ) -> None:
        self._lock = Lock()
        self._acquisition_attempts = acquisition_attempts
        self._observed_thread_name = observed_thread_name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if current_thread().name == self._observed_thread_name:
            self._acquisition_attempts.put("model-lock")
        if not blocking:
            return self._lock.acquire(blocking=False)
        if timeout == -1:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()


def _start_thread(
    *, name: str, target: Callable[[], None], errors: List[BaseException]
) -> Thread:
    def run() -> None:
        try:
            target()
        except BaseException as error:
            errors.append(error)

    thread = Thread(name=name, target=run)
    thread.start()
    return thread


def _wait(event: Event, message: str) -> None:
    assert event.wait(timeout=THREAD_TIMEOUT), message


def _join_threads(threads: List[Thread]) -> None:
    for thread in threads:
        thread.join(timeout=THREAD_TIMEOUT)
        assert not thread.is_alive(), f"Thread {thread.name} did not finish"


def _assert_quiescent_model_entry(model_manager: ModelManager, model_id: str) -> None:
    assert model_id in model_manager.models()
    assert model_manager._model_lock_entries[model_id].users == 0


def test_model_manager_decorator_records_request_metadata_for_warm_model() -> None:
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager._models = {"sam3/sam3_final": MagicMock()}
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
    assert description.model_id == "sam3/sam3_final"
    assert description.request_aliases == ["some/1"]
    assert description.request_paths == ["/sam3/concept_segment"]
    assert ids.get_ids() == {"sam3/sam3_final"}


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


@pytest.mark.parametrize(
    "decorate",
    [
        pytest.param(ModelManagerDecorator, id="base-decorator"),
        pytest.param(
            lambda manager: WithFixedSizeCache(manager, max_size=8),
            id="fixed-size-cache",
        ),
    ],
)
def test_warm_add_preserves_an_established_remove_mutation_order(
    monkeypatch: pytest.MonkeyPatch,
    decorate: Callable[[ModelManager], ModelManagerDecorator],
) -> None:
    removal_started = Event()
    removal_finished = Event()
    allow_removal_to_finish = Event()
    add_returned = Event()
    add_progress: Queue = Queue()

    class LoadedModel:
        task_type = "loaded"

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            removal_started.set()
            if not allow_removal_to_finish.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release model removal")

    class ReplacementModel:
        task_type = "replacement"

        def __init__(self, **kwargs) -> None:
            pass

    model_registry = MagicMock()
    model_registry.get_model.return_value = ReplacementModel
    base_manager = ModelManager(
        model_registry=model_registry,
        models={MODEL_ID: LoadedModel()},
    )
    manager = decorate(base_manager)
    original_record_request_metadata = base_manager.record_request_metadata

    def record_request_metadata(*args, **kwargs) -> None:
        if current_thread().name == "add-model":
            add_progress.put("metadata")
        original_record_request_metadata(*args, **kwargs)

    monkeypatch.setattr(
        base_manager, "record_request_metadata", record_request_metadata
    )
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(acquisition_attempts=add_progress),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    remove_errors: List[BaseException] = []
    add_errors: List[BaseException] = []

    def remove_model() -> None:
        try:
            manager.remove(MODEL_ID)
        finally:
            removal_finished.set()

    def add_model() -> None:
        manager.add_model(MODEL_ID, "api-key")
        add_returned.set()

    threads = [
        _start_thread(
            name="remove-model",
            target=remove_model,
            errors=remove_errors,
        )
    ]
    try:
        _wait(removal_started, "remove() did not start clearing the loaded model")
        threads.append(
            _start_thread(
                name="add-model",
                target=add_model,
                errors=add_errors,
            )
        )
        add_progress.get(timeout=THREAD_TIMEOUT)
        assert not add_returned.wait(timeout=MUTATION_ORDER_OBSERVATION_TIMEOUT)
        assert not removal_finished.is_set()
    finally:
        allow_removal_to_finish.set()
        _join_threads(threads)

    assert remove_errors == []
    assert add_errors == []
    assert removal_finished.is_set()
    assert add_returned.is_set()
    assert isinstance(base_manager.models()[MODEL_ID], ReplacementModel)
    assert model_registry.get_model.call_count == 1
    _assert_quiescent_model_entry(base_manager, MODEL_ID)
    if isinstance(manager, WithFixedSizeCache):
        assert list(manager._key_queue) == [MODEL_ID]


@pytest.mark.parametrize("operation", ["remove", "evict"])
def test_fixed_size_cache_restores_queue_when_model_removal_fails(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    class FailingRemovalModel:
        task_type = "test"

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            raise RuntimeError("model removal failed")

    loaded_model = FailingRemovalModel()
    base_manager = ModelManager(
        model_registry=MagicMock(), models={MODEL_ID: loaded_model}
    )
    manager = WithFixedSizeCache(base_manager, max_size=1)
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    with pytest.raises(RuntimeError, match="model removal failed"):
        if operation == "remove":
            manager.remove(MODEL_ID)
        else:
            manager.add_model("other/1", "api-key")

    assert base_manager.models() == {MODEL_ID: loaded_model}
    assert list(manager._key_queue) == [MODEL_ID]
    _assert_quiescent_model_entry(base_manager, MODEL_ID)


def test_fixed_size_cache_failed_direct_remove_preserves_exact_lru_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ids = ["older/1", MODEL_ID, "newer/1"]

    class LoadedModel:
        task_type = "test"

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            if self.model_id == MODEL_ID:
                raise RuntimeError("model removal failed")

    loaded_models = {model_id: LoadedModel(model_id=model_id) for model_id in model_ids}
    base_manager = ModelManager(model_registry=MagicMock(), models=loaded_models)
    manager = WithFixedSizeCache(base_manager, max_size=8)
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    with pytest.raises(RuntimeError, match="model removal failed"):
        manager.remove(MODEL_ID)

    assert base_manager.models() == loaded_models
    assert list(manager._key_queue) == model_ids
    _assert_quiescent_model_entry(base_manager, MODEL_ID)


def test_fixed_size_cache_failed_eviction_restores_skipped_pinned_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ids = ["pinned-a/1", "pinned-b/1", MODEL_ID, "resident/1"]

    class LoadedModel:
        task_type = "test"

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            if self.model_id == MODEL_ID:
                raise RuntimeError("model eviction failed")

    loaded_models = {model_id: LoadedModel(model_id=model_id) for model_id in model_ids}
    base_manager = ModelManager(model_registry=MagicMock(), models=loaded_models)
    manager = WithFixedSizeCache(base_manager, max_size=len(model_ids))
    manager.pin_model("pinned-a/1")
    manager.pin_model("pinned-b/1")
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    with pytest.raises(RuntimeError, match="model eviction failed"):
        manager.add_model("new/1", "api-key")

    assert base_manager.models() == loaded_models
    assert list(manager._key_queue) == model_ids
    _assert_quiescent_model_entry(base_manager, MODEL_ID)


def test_fixed_size_cache_failed_multi_eviction_keeps_only_registered_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ids = ["evicted/1", MODEL_ID, "resident/1"]
    cleared_models = []

    class LoadedModel:
        task_type = "test"

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            cleared_models.append(self.model_id)
            if self.model_id == MODEL_ID:
                raise RuntimeError("second eviction failed")

    loaded_models = {model_id: LoadedModel(model_id=model_id) for model_id in model_ids}
    base_manager = ModelManager(model_registry=MagicMock(), models=loaded_models)
    manager = WithFixedSizeCache(base_manager, max_size=len(model_ids))
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    with pytest.raises(RuntimeError, match="second eviction failed"):
        manager.add_model("new/1", "api-key")

    assert cleared_models == ["evicted/1", MODEL_ID]
    assert set(base_manager.models()) == {MODEL_ID, "resident/1"}
    assert list(manager._key_queue) == [MODEL_ID, "resident/1"]
    assert "evicted/1" not in base_manager._model_lock_entries
    _assert_quiescent_model_entry(base_manager, MODEL_ID)


def test_fixed_size_cache_keeps_one_queue_entry_for_concurrent_cold_adds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization_started = Event()
    allow_initialization_to_finish = Event()
    second_add_lock_attempts: Queue = Queue()

    class BlockingModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            initialization_started.set()
            if not allow_initialization_to_finish.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release model initialization")

    model_registry = MagicMock()
    model_registry.get_model.return_value = BlockingModel
    base_manager = ModelManager(model_registry=model_registry)
    manager = WithFixedSizeCache(base_manager, max_size=8)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts=second_add_lock_attempts,
            observed_thread_name="second-add",
        ),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="first-add",
            target=lambda: manager.add_model(MODEL_ID, "api-key"),
            errors=errors,
        )
    ]
    try:
        _wait(initialization_started, "The first model load did not start")
        threads.append(
            _start_thread(
                name="second-add",
                target=lambda: manager.add_model(MODEL_ID, "api-key"),
                errors=errors,
            )
        )
        second_add_lock_attempts.get(timeout=THREAD_TIMEOUT)
    finally:
        allow_initialization_to_finish.set()
        _join_threads(threads)

    assert errors == []
    assert model_registry.get_model.call_count == 1
    assert MODEL_ID in base_manager.models()
    assert list(manager._key_queue) == [MODEL_ID]
    _assert_quiescent_model_entry(base_manager, MODEL_ID)


def test_fixed_size_cache_loads_and_tracks_the_resolved_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Model:
        task_type = "test"

        def __init__(self, model_id: str = MODEL_ID, **kwargs) -> None:
            self.model_id = model_id

    model_registry = MagicMock()
    model_registry.get_model.return_value = Model
    base_manager = ModelManager(
        model_registry=model_registry,
        models={MODEL_ID: Model()},
    )
    manager = WithFixedSizeCache(base_manager, max_size=8)
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    manager.add_model(MODEL_ID, "api-key", model_id_alias=ALIAS_ID)

    assert set(base_manager.models()) == {MODEL_ID, ALIAS_ID}
    assert list(manager._key_queue) == [MODEL_ID, ALIAS_ID]
    model_registry.get_model.assert_called_once()
    _assert_quiescent_model_entry(base_manager, ALIAS_ID)


def test_fixed_size_cache_eviction_and_reload_keep_registry_and_queue_consistent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eviction_started = Event()
    allow_eviction_to_finish = Event()
    reload_lock_attempts: Queue = Queue()

    class LoadedModel:
        task_type = "loaded"

        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            if self.model_id == MODEL_ID:
                eviction_started.set()
                if not allow_eviction_to_finish.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release cache eviction")

    class ReplacementModel:
        task_type = "replacement"

        def __init__(self, model_id: str, **kwargs) -> None:
            self.model_id = model_id

    model_registry = MagicMock()
    model_registry.get_model.return_value = ReplacementModel
    base_manager = ModelManager(
        model_registry=model_registry,
        models={
            MODEL_ID: LoadedModel(MODEL_ID),
            "resident/1": LoadedModel("resident/1"),
        },
    )
    manager = WithFixedSizeCache(base_manager, max_size=2)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts=reload_lock_attempts,
            observed_thread_name="reload-evicted-model",
        ),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="evict-model",
            target=lambda: manager.add_model("new/1", "api-key"),
            errors=errors,
        )
    ]
    try:
        _wait(eviction_started, "Adding the new model did not start eviction")
        threads.append(
            _start_thread(
                name="reload-evicted-model",
                target=lambda: manager.add_model(MODEL_ID, "api-key"),
                errors=errors,
            )
        )
        reload_lock_attempts.get(timeout=THREAD_TIMEOUT)
    finally:
        allow_eviction_to_finish.set()
        _join_threads(threads)

    assert errors == []
    assert set(base_manager.models()) == {MODEL_ID, "new/1"}
    assert len(manager._key_queue) == 2
    assert set(manager._key_queue) == set(base_manager.models())
    assert model_registry.get_model.call_count == 2
    assert set(base_manager._model_lock_entries) == set(base_manager.models())
    assert all(entry.users == 0 for entry in base_manager._model_lock_entries.values())
