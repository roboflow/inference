from contextlib import contextmanager
from threading import Event, Lock, Thread, current_thread
from typing import Callable, Dict, Generator, List
from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import ModelManagerLockAcquisitionError
from inference.core.managers import base as base_module
from inference.core.managers.base import ModelManager
from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache

MODEL_ID = "example/1"
ALIAS_ID = "alias/1"
THREAD_TIMEOUT = 2.0
CONCURRENT_INITIALIZATION_TIMEOUT = 0.5


class _ObservableLock:
    def __init__(self, acquisition_attempts: Dict[str, Event]) -> None:
        self._lock = Lock()
        self._acquisition_attempts = acquisition_attempts

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        attempt_event = self._acquisition_attempts.get(current_thread().name)
        if attempt_event is not None:
            attempt_event.set()
        if not blocking:
            return self._lock.acquire(blocking=False)
        if timeout == -1:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()


class _SequencedObservableLock:
    def __init__(self, thread_name: str, acquisition_events: List[Event]) -> None:
        self._lock = Lock()
        self._thread_name = thread_name
        self._acquisition_events = acquisition_events
        self._target_acquisitions = 0

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if current_thread().name == self._thread_name:
            self._target_acquisitions += 1
            event_index = self._target_acquisitions - 1
            if event_index < len(self._acquisition_events):
                self._acquisition_events[event_index].set()
        if not blocking:
            return self._lock.acquire(blocking=False)
        if timeout == -1:
            return self._lock.acquire()
        return self._lock.acquire(timeout=timeout)

    def release(self) -> None:
        self._lock.release()

    def __enter__(self) -> "_SequencedObservableLock":
        self.acquire()
        return self

    def __exit__(self, *args) -> None:
        self.release()


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


def _assert_quiescent_model_lock_registry(model_manager: ModelManager) -> None:
    assert set(model_manager._model_lock_entries) == set(model_manager.models())
    assert all(entry.users == 0 for entry in model_manager._model_lock_entries.values())


def _assert_model_lock_generation_retired(
    model_manager: ModelManager, model_id: str
) -> None:
    assert model_id not in model_manager.models()
    assert model_id not in model_manager._model_lock_entries


def test_concurrent_adds_initialize_same_model_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_initialization_started = Event()
    allow_initialization_to_finish = Event()
    second_acquisition_attempted = Event()
    counter_lock = Lock()
    initialization_count = 0

    class BlockingModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            nonlocal initialization_count
            with counter_lock:
                initialization_count += 1
            first_initialization_started.set()
            if not allow_initialization_to_finish.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release model initialization")

    model_registry = MagicMock()
    model_registry.get_model.return_value = BlockingModel
    model_manager = ModelManager(model_registry=model_registry)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts={"second-add": second_acquisition_attempted}
        ),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="first-add",
            target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
            errors=errors,
        )
    ]
    try:
        _wait(first_initialization_started, "The first model load did not start")
        threads.append(
            _start_thread(
                name="second-add",
                target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                errors=errors,
            )
        )
        _wait(
            second_acquisition_attempted,
            "The second add_model() did not attempt the per-model lock",
        )
        assert initialization_count == 1
    finally:
        allow_initialization_to_finish.set()
        _join_threads(threads)

    assert errors == []
    assert initialization_count == 1
    assert model_registry.get_model.call_count == 1
    _assert_quiescent_model_lock_registry(model_manager)


def test_timed_out_waiter_does_not_retire_active_lock_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_initialization_started = Event()
    allow_initialization_to_finish = Event()
    waiter_acquisition_attempted = Event()

    class BlockingModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            first_initialization_started.set()
            if not allow_initialization_to_finish.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release model initialization")

    model_registry = MagicMock()
    model_registry.get_model.return_value = BlockingModel
    model_manager = ModelManager(model_registry=model_registry)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts={"timed-out-add": waiter_acquisition_attempted}
        ),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)
    original_acquire_with_timeout = base_module.acquire_with_timeout

    @contextmanager
    def acquire_with_short_model_timeout(
        lock: Lock, timeout: float = THREAD_TIMEOUT
    ) -> Generator[bool, None, None]:
        effective_timeout = (
            THREAD_TIMEOUT if lock is model_manager._state_lock else 0.05
        )
        with original_acquire_with_timeout(
            lock=lock, timeout=effective_timeout
        ) as acquired:
            yield acquired

    monkeypatch.setattr(
        base_module, "acquire_with_timeout", acquire_with_short_model_timeout
    )

    first_errors: List[BaseException] = []
    waiter_errors: List[BaseException] = []
    first_thread = _start_thread(
        name="first-add",
        target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
        errors=first_errors,
    )
    try:
        _wait(first_initialization_started, "The first model load did not start")
        active_generation = model_manager._model_lock_entries[MODEL_ID]
        waiter_thread = _start_thread(
            name="timed-out-add",
            target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
            errors=waiter_errors,
        )
        _wait(
            waiter_acquisition_attempted,
            "The timed-out caller did not attempt the per-model lock",
        )
        _join_threads([waiter_thread])

        assert len(waiter_errors) == 1
        assert isinstance(waiter_errors[0], ModelManagerLockAcquisitionError)
        assert model_manager._model_lock_entries[MODEL_ID] is active_generation
        assert active_generation.users == 1
    finally:
        allow_initialization_to_finish.set()
        _join_threads([first_thread])

    assert first_errors == []
    assert model_manager._model_lock_entries[MODEL_ID] is active_generation
    _assert_quiescent_model_lock_registry(model_manager)


def test_fixed_size_cache_load_error_keeps_generation_for_multiple_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failing_initialization_started = Event()
    allow_failure = Event()
    successful_initialization_started = Event()
    concurrent_initialization_started = Event()
    allow_successful_initialization_to_finish = Event()
    counter_lock = Lock()
    constructor_calls = 0
    active_successful_initializations = 0
    max_active_successful_initializations = 0
    waiter_names = [f"waiting-add-{index}" for index in range(3)]
    acquisition_attempts = {name: Event() for name in waiter_names}
    acquisition_attempts["new-add"] = Event()

    class FailsOnceModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            nonlocal constructor_calls
            nonlocal active_successful_initializations
            nonlocal max_active_successful_initializations
            with counter_lock:
                constructor_calls += 1
                current_call = constructor_calls
                if current_call > 1:
                    active_successful_initializations += 1
                    max_active_successful_initializations = max(
                        max_active_successful_initializations,
                        active_successful_initializations,
                    )
                    if active_successful_initializations == 1:
                        successful_initialization_started.set()
                    else:
                        concurrent_initialization_started.set()
            if current_call == 1:
                failing_initialization_started.set()
                if not allow_failure.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release the failing load")
                raise RuntimeError("model load failed")
            try:
                if not allow_successful_initialization_to_finish.wait(
                    timeout=THREAD_TIMEOUT
                ):
                    raise AssertionError("Test did not release the successful load")
            finally:
                with counter_lock:
                    active_successful_initializations -= 1

    model_registry = MagicMock()
    model_registry.get_model.return_value = FailsOnceModel
    base_model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(base_model_manager, max_size=8)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(acquisition_attempts=acquisition_attempts),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    failing_errors: List[BaseException] = []
    waiter_errors: List[BaseException] = []
    newcomer_errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="failing-add",
            target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
            errors=failing_errors,
        )
    ]
    concurrent_initialization_observed = False
    try:
        _wait(failing_initialization_started, "The failing model load did not start")
        for waiter_name in waiter_names:
            threads.append(
                _start_thread(
                    name=waiter_name,
                    target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                    errors=waiter_errors,
                )
            )
        for waiter_name in waiter_names:
            _wait(
                acquisition_attempts[waiter_name],
                f"{waiter_name} did not attempt the first lock generation",
            )

        allow_failure.set()
        _join_threads([threads[0]])
        _wait(
            successful_initialization_started,
            "No waiter retried model initialization after the load error",
        )

        threads.append(
            _start_thread(
                name="new-add",
                target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                errors=newcomer_errors,
            )
        )
        _wait(
            acquisition_attempts["new-add"],
            "The newcomer did not attempt the current lock generation",
        )
        concurrent_initialization_observed = concurrent_initialization_started.wait(
            timeout=CONCURRENT_INITIALIZATION_TIMEOUT
        )
    finally:
        allow_failure.set()
        allow_successful_initialization_to_finish.set()
        _join_threads(threads)

    assert len(failing_errors) == 1
    assert isinstance(failing_errors[0], RuntimeError)
    assert str(failing_errors[0]) == "model load failed"
    assert waiter_errors == []
    assert newcomer_errors == []
    assert not concurrent_initialization_observed
    assert max_active_successful_initializations == 1
    assert constructor_calls == 2
    assert MODEL_ID in model_manager.models()
    assert list(model_manager._key_queue) == [MODEL_ID]
    _assert_quiescent_model_lock_registry(base_model_manager)


def test_load_errors_retire_generation_after_all_waiters_finish(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_initialization_started = Event()
    allow_first_initialization_to_fail = Event()
    waiter_names = ["failing-waiter-a", "failing-waiter-b"]
    acquisition_attempts = {name: Event() for name in waiter_names}
    constructor_calls = 0

    class AlwaysFailingModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            nonlocal constructor_calls
            constructor_calls += 1
            if constructor_calls == 1:
                first_initialization_started.set()
                if not allow_first_initialization_to_fail.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release the first failed load")
            raise RuntimeError("model load failed")

    model_registry = MagicMock()
    model_registry.get_model.return_value = AlwaysFailingModel
    base_model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(base_model_manager, max_size=8)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(acquisition_attempts=acquisition_attempts),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="first-failing-add",
            target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
            errors=errors,
        )
    ]
    try:
        _wait(
            first_initialization_started,
            "The first failing model load did not start",
        )
        generation = base_model_manager._model_lock_entries[MODEL_ID]
        for waiter_name in waiter_names:
            threads.append(
                _start_thread(
                    name=waiter_name,
                    target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                    errors=errors,
                )
            )
        for waiter_name in waiter_names:
            _wait(
                acquisition_attempts[waiter_name],
                f"{waiter_name} did not wait on the original generation",
            )
    finally:
        allow_first_initialization_to_fail.set()
        _join_threads(threads)

    assert len(errors) == 3
    assert all(isinstance(error, RuntimeError) for error in errors)
    assert all(str(error) == "model load failed" for error in errors)
    assert constructor_calls == 3
    assert generation.users == 0
    assert list(model_manager._key_queue) == []
    _assert_model_lock_generation_retired(base_model_manager, MODEL_ID)


def test_load_error_cleanup_waits_for_registry_lock_and_preserves_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initialization_started = Event()
    allow_initialization_to_fail = Event()
    initial_registry_acquisition = Event()
    cleanup_registry_acquisition = Event()

    class FailingModel:
        task_type = "test"

        def __init__(self, **kwargs) -> None:
            initialization_started.set()
            if not allow_initialization_to_fail.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release the failing model load")
            raise RuntimeError("original model load failure")

    model_registry = MagicMock()
    model_registry.get_model.return_value = FailingModel
    model_manager = ModelManager(model_registry=model_registry)
    state_lock = _SequencedObservableLock(
        thread_name="failing-add",
        acquisition_events=[
            initial_registry_acquisition,
            cleanup_registry_acquisition,
        ],
    )
    model_manager._state_lock = state_lock
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    errors: List[BaseException] = []
    thread = _start_thread(
        name="failing-add",
        target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
        errors=errors,
    )
    state_lock_held = False
    try:
        _wait(
            initial_registry_acquisition,
            "The caller did not reserve its model lock generation",
        )
        _wait(initialization_started, "The failing model load did not start")
        generation = model_manager._model_lock_entries[MODEL_ID]
        state_lock_held = state_lock.acquire(timeout=THREAD_TIMEOUT)
        assert state_lock_held
        allow_initialization_to_fail.set()
        _wait(
            cleanup_registry_acquisition,
            "The caller did not wait to clean up its model lock reservation",
        )
        assert thread.is_alive()
        assert errors == []
    finally:
        allow_initialization_to_fail.set()
        if state_lock_held:
            state_lock.release()
        _join_threads([thread])

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert str(errors[0]) == "original model load failure"
    assert generation.users == 0
    _assert_model_lock_generation_retired(model_manager, MODEL_ID)


def test_remove_error_releases_lock_without_retiring_loaded_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    removal_started = Event()
    allow_removal_to_fail = Event()
    waiter_acquisition_attempted = Event()

    class FailsOnceOnRemoval:
        task_type = "test"

        def __init__(self) -> None:
            self.removal_calls = 0

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            self.removal_calls += 1
            if self.removal_calls == 1:
                removal_started.set()
                if not allow_removal_to_fail.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release the failing removal")
                raise RuntimeError("model removal failed")

    loaded_model = FailsOnceOnRemoval()
    model_registry = MagicMock()
    model_manager = ModelManager(
        model_registry=model_registry, models={MODEL_ID: loaded_model}
    )
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts={"waiting-add": waiter_acquisition_attempted}
        ),
    )
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    remove_errors: List[BaseException] = []
    add_errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="remove",
            target=lambda: model_manager.remove(MODEL_ID),
            errors=remove_errors,
        )
    ]
    try:
        _wait(removal_started, "The failing removal did not start")
        threads.append(
            _start_thread(
                name="waiting-add",
                target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                errors=add_errors,
            )
        )
        _wait(
            waiter_acquisition_attempted,
            "The waiting add_model() did not attempt the per-model lock",
        )
        allow_removal_to_fail.set()
    finally:
        allow_removal_to_fail.set()
        _join_threads(threads)

    assert len(remove_errors) == 1
    assert isinstance(remove_errors[0], RuntimeError)
    assert str(remove_errors[0]) == "model removal failed"
    assert add_errors == []
    assert model_registry.get_model.call_count == 0
    assert model_manager.models()[MODEL_ID] is loaded_model
    _assert_quiescent_model_lock_registry(model_manager)

    model_manager.remove(MODEL_ID)

    _assert_model_lock_generation_retired(model_manager, MODEL_ID)


def test_alias_lifecycle_reuses_then_retires_one_resolved_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_initialization_started = Event()
    allow_first_initialization_to_finish = Event()
    second_acquisition_attempted = Event()
    constructed_model_ids: List[str] = []

    class LoadedModel:
        task_type = "test"

        def __init__(self, model_id: str, **kwargs) -> None:
            self.model_id = model_id
            constructed_model_ids.append(model_id)
            if model_id == "original-a/1":
                first_initialization_started.set()
                if not allow_first_initialization_to_finish.wait(
                    timeout=THREAD_TIMEOUT
                ):
                    raise AssertionError("Test did not release alias initialization")

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            pass

    model_registry = MagicMock()
    model_registry.get_model.return_value = LoadedModel
    base_model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(base_model_manager, max_size=8)
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(
            acquisition_attempts={"alias-add-b": second_acquisition_attempted}
        ),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)
    monkeypatch.setattr(base_module, "try_releasing_cuda_memory", lambda: None)

    errors: List[BaseException] = []
    threads = [
        _start_thread(
            name="alias-add-a",
            target=lambda: model_manager.add_model(
                "original-a/1", "api-key", model_id_alias=ALIAS_ID
            ),
            errors=errors,
        )
    ]
    try:
        _wait(
            first_initialization_started,
            "The first aliased model initialization did not start",
        )
        first_generation = base_model_manager._model_lock_entries[ALIAS_ID]
        threads.append(
            _start_thread(
                name="alias-add-b",
                target=lambda: model_manager.add_model(
                    "original-b/1", "api-key", model_id_alias=ALIAS_ID
                ),
                errors=errors,
            )
        )
        _wait(
            second_acquisition_attempted,
            "The second aliased caller did not attempt the resolved lock",
        )
    finally:
        allow_first_initialization_to_finish.set()
        _join_threads(threads)

    assert errors == []
    assert constructed_model_ids == ["original-a/1"]
    assert base_model_manager.models()[ALIAS_ID].model_id == "original-a/1"
    assert base_model_manager._model_lock_entries[ALIAS_ID] is first_generation
    assert list(model_manager._key_queue) == [ALIAS_ID]
    _assert_quiescent_model_lock_registry(base_model_manager)

    model_manager.remove(ALIAS_ID)

    assert list(model_manager._key_queue) == []
    _assert_model_lock_generation_retired(base_model_manager, ALIAS_ID)

    model_manager.add_model("original-c/1", "api-key", model_id_alias=ALIAS_ID)
    second_generation = base_model_manager._model_lock_entries[ALIAS_ID]

    assert second_generation is not first_generation
    assert constructed_model_ids == ["original-a/1", "original-c/1"]
    assert base_model_manager.models()[ALIAS_ID].model_id == "original-c/1"
    assert list(model_manager._key_queue) == [ALIAS_ID]
    _assert_quiescent_model_lock_registry(base_model_manager)


def test_fixed_size_cache_does_not_serialize_different_resolved_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_model_started = Event()
    allow_first_model_to_finish = Event()
    second_model_finished = Event()

    class Model:
        task_type = "test"

        def __init__(self, model_id: str, **kwargs) -> None:
            if model_id == "model-a/1":
                first_model_started.set()
                if not allow_first_model_to_finish.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release the first model")

    model_registry = MagicMock()
    model_registry.get_model.return_value = Model
    base_model_manager = ModelManager(model_registry=model_registry)
    model_manager = WithFixedSizeCache(base_model_manager, max_size=8)
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    first_errors: List[BaseException] = []
    second_errors: List[BaseException] = []
    first_thread = _start_thread(
        name="add-model-a",
        target=lambda: model_manager.add_model("model-a/1", "api-key"),
        errors=first_errors,
    )
    threads = [first_thread]

    def add_second_model() -> None:
        model_manager.add_model("model-b/1", "api-key")
        second_model_finished.set()

    try:
        _wait(first_model_started, "The first model load did not start")
        threads.append(
            _start_thread(
                name="add-model-b", target=add_second_model, errors=second_errors
            )
        )
        _wait(
            second_model_finished,
            "A different resolved identifier was blocked by the first model load",
        )
    finally:
        allow_first_model_to_finish.set()
        _join_threads(threads)

    assert first_errors == []
    assert second_errors == []
    assert set(model_manager.models()) == {"model-a/1", "model-b/1"}
    assert len(model_manager._key_queue) == 2
    assert set(model_manager._key_queue) == {"model-a/1", "model-b/1"}
    _assert_quiescent_model_lock_registry(base_model_manager)


def test_model_lock_generation_survives_holder_and_waiters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remove_started = Event()
    allow_remove_to_delete_model = Event()
    remove_paused_before_model_lock_release = Event()
    allow_model_lock_release = Event()
    waiter_acquisition_attempted = Event()
    newcomer_acquisition_attempted = Event()
    first_initialization_started = Event()
    concurrent_initialization_started = Event()
    allow_initializations_to_finish = Event()
    counter_lock = Lock()
    active_initializations = 0
    max_active_initializations = 0

    class OldModel:
        task_type = "old"

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            remove_started.set()
            if not allow_remove_to_delete_model.wait(timeout=THREAD_TIMEOUT):
                raise AssertionError("Test did not release the removal operation")

    class NewModel:
        task_type = "new"

        def __init__(self, **kwargs) -> None:
            nonlocal active_initializations, max_active_initializations
            with counter_lock:
                active_initializations += 1
                max_active_initializations = max(
                    max_active_initializations, active_initializations
                )
                if active_initializations == 1:
                    first_initialization_started.set()
                else:
                    concurrent_initialization_started.set()
            try:
                if not allow_initializations_to_finish.wait(timeout=THREAD_TIMEOUT):
                    raise AssertionError("Test did not release model initialization")
            finally:
                with counter_lock:
                    active_initializations -= 1

        def clear_cache(self, delete_from_disk: bool = True) -> None:
            pass

    model_registry = MagicMock()
    model_registry.get_model.return_value = NewModel
    model_manager = ModelManager(model_registry=model_registry)
    model_manager._models[MODEL_ID] = OldModel()

    acquisition_attempts = {
        "waiting-add": waiter_acquisition_attempted,
        "new-add": newcomer_acquisition_attempted,
    }
    monkeypatch.setattr(
        base_module,
        "Lock",
        lambda: _ObservableLock(acquisition_attempts=acquisition_attempts),
    )
    monkeypatch.setattr(base_module, "_get_cuda_memory_allocated", lambda: None)

    def pause_remove_before_model_lock_release() -> None:
        remove_paused_before_model_lock_release.set()
        if not allow_model_lock_release.wait(timeout=THREAD_TIMEOUT):
            raise AssertionError("Test did not release the per-model lock holder")

    monkeypatch.setattr(
        base_module,
        "try_releasing_cuda_memory",
        pause_remove_before_model_lock_release,
    )

    errors: List[BaseException] = []
    threads: List[Thread] = []
    concurrent_initialization_observed = False
    try:
        threads.append(
            _start_thread(
                name="remove",
                target=lambda: model_manager.remove(MODEL_ID),
                errors=errors,
            )
        )
        _wait(remove_started, "remove() did not acquire the first lock generation")

        threads.append(
            _start_thread(
                name="waiting-add",
                target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                errors=errors,
            )
        )
        _wait(
            waiter_acquisition_attempted,
            "The waiting add_model() did not attempt to acquire the first generation",
        )

        allow_remove_to_delete_model.set()
        _wait(
            remove_paused_before_model_lock_release,
            "remove() did not reach the post-disposal, pre-release checkpoint",
        )

        threads.append(
            _start_thread(
                name="new-add",
                target=lambda: model_manager.add_model(MODEL_ID, "api-key"),
                errors=errors,
            )
        )
        _wait(
            newcomer_acquisition_attempted,
            "The newcomer add_model() did not attempt to acquire a lock",
        )

        allow_model_lock_release.set()
        _wait(first_initialization_started, "No replacement model was initialized")
        concurrent_initialization_observed = concurrent_initialization_started.wait(
            timeout=CONCURRENT_INITIALIZATION_TIMEOUT
        )
    finally:
        allow_remove_to_delete_model.set()
        allow_model_lock_release.set()
        allow_initializations_to_finish.set()
        _join_threads(threads)

    assert errors == []
    assert not concurrent_initialization_observed
    assert max_active_initializations == 1
    assert MODEL_ID in model_manager.models()
    _assert_quiescent_model_lock_registry(model_manager)

    model_manager.remove(MODEL_ID)

    _assert_model_lock_generation_retired(model_manager, MODEL_ID)
