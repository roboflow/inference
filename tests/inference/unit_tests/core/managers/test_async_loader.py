"""
Tests for AsyncModelLoader (Issue #2448)

Tests the async model loading infrastructure that prevents workflow thread pool
starvation by isolating model loads to a dedicated executor.
"""
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from inference.core.managers.async_loader import AsyncModelLoader, get_global_async_loader
from inference.core.managers.base import ModelManager


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_id: str, api_key: str, **kwargs):
        self.model_id = model_id
        self.api_key = api_key
        self.task_type = "object-detection"
        self._vram_bytes = 1000

    def clear_cache(self, delete_from_disk: bool = True):
        pass


class TestAsyncModelLoader:
    """Test AsyncModelLoader functionality."""

    def test_model_already_loaded_returns_none(self):
        """Test that async load returns None if model is already loaded."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        manager.add_model(model_id="test/1", api_key="key")

        loader = AsyncModelLoader(max_workers=2)

        # when
        future = loader.add_model_async(
            model_manager=manager,
            model_id="test/1",
            api_key="key",
        )

        # then
        assert future is None
        loader.shutdown(wait=True)

    def test_cold_model_returns_future(self):
        """Test that async load returns Future for cold (not loaded) model."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when
        future = loader.add_model_async(
            model_manager=manager,
            model_id="test/1",
            api_key="key",
        )

        # then
        assert future is not None
        assert not future.done()

        # Wait for completion
        future.result(timeout=5.0)
        assert "test/1" in manager

        loader.shutdown(wait=True)

    def test_concurrent_loads_are_coalesced(self):
        """Test that multiple concurrent requests for same model are coalesced."""
        # given
        model_registry = MagicMock()

        # Make model loading slow to ensure concurrency
        load_count = {"count": 0}
        load_started = threading.Event()

        def slow_mock_model(*args, **kwargs):
            load_count["count"] += 1
            load_started.set()
            time.sleep(0.5)  # Simulate slow load
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = slow_mock_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when - start 3 concurrent requests for the same model
        def load_model():
            future = loader.add_model_async(
                model_manager=manager,
                model_id="test/1",
                api_key="key",
            )
            if future:
                future.result(timeout=5.0)

        threads = [threading.Thread(target=load_model) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        # then - model should only be loaded once (coalesced)
        assert load_count["count"] == 1
        assert "test/1" in manager

        # Check stats
        stats = loader.get_stats()
        assert stats["pending_loads"] == 0  # Should be cleaned up

        loader.shutdown(wait=True)

    def test_load_failure_propagates_to_all_waiters(self):
        """Test that if a load fails, all waiters get the exception."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.side_effect = RuntimeError("Load failed!")

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when
        future1 = loader.add_model_async(manager, "test/1", "key")
        future2 = loader.add_model_async(manager, "test/1", "key")

        # then - both futures should raise the same exception
        with pytest.raises(RuntimeError, match="Load failed"):
            future1.result(timeout=5.0)

        with pytest.raises(RuntimeError, match="Load failed"):
            future2.result(timeout=5.0)

        loader.shutdown(wait=True)

    def test_different_models_load_in_parallel(self):
        """Test that different models can load in parallel."""
        # given
        model_registry = MagicMock()

        load_times = {}
        load_started = {}

        def timed_mock_model(model_id, *args, **kwargs):
            load_started[model_id] = time.time()
            time.sleep(0.3)  # Simulate load time
            load_times[model_id] = time.time()
            return MockModel(model_id, *args, **kwargs)

        model_registry.get_model.return_value = timed_mock_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=4)

        # when - start loads for 3 different models
        futures = []
        for i in range(3):
            future = loader.add_model_async(
                manager, f"test/{i}", "key"
            )
            futures.append(future)

        # Wait for all
        for f in futures:
            f.result(timeout=5.0)

        # then - all models should be loaded
        assert "test/0" in manager
        assert "test/1" in manager
        assert "test/2" in manager

        # Check that they loaded in parallel (overlapping time ranges)
        # If sequential, would take 0.9s+. If parallel, should be ~0.3s
        total_time = max(load_times.values()) - min(load_started.values())
        assert total_time < 0.6  # Should be close to 0.3s, not 0.9s

        loader.shutdown(wait=True)

    def test_stats_tracking(self):
        """Test that loader stats are tracked correctly."""
        # given
        model_registry = MagicMock()

        loading = threading.Event()
        can_finish = threading.Event()

        def blocking_model(*args, **kwargs):
            loading.set()
            can_finish.wait(timeout=5.0)
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = blocking_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when - start a load and check stats while it's in progress
        future = loader.add_model_async(manager, "test/1", "key")
        loading.wait(timeout=1.0)

        stats = loader.get_stats()

        # then
        assert stats["pending_loads"] == 1
        assert stats["total_waiters"] == 1
        assert stats["max_workers"] == 2

        # Clean up
        can_finish.set()
        future.result(timeout=5.0)
        loader.shutdown(wait=True)

    def test_global_loader_singleton(self):
        """Test that get_global_async_loader returns a singleton."""
        # when
        loader1 = get_global_async_loader()
        loader2 = get_global_async_loader()

        # then
        assert loader1 is loader2

    def test_timeout_on_slow_load(self):
        """Test that Future.result() respects timeout."""
        # given
        model_registry = MagicMock()

        def very_slow_model(*args, **kwargs):
            time.sleep(10.0)  # Very slow
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = very_slow_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=1)

        # when
        future = loader.add_model_async(manager, "test/1", "key")

        # then - should timeout
        with pytest.raises(Exception):  # TimeoutError or concurrent.futures.TimeoutError
            future.result(timeout=0.1)

        loader.shutdown(wait=False)  # Don't wait for the slow load

    def test_load_cleanup_after_completion(self):
        """Test that pending loads are cleaned up after completion."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when
        future = loader.add_model_async(manager, "test/1", "key")
        future.result(timeout=5.0)

        # then - should be cleaned up
        stats = loader.get_stats()
        assert stats["pending_loads"] == 0
        assert stats["total_waiters"] == 0

        loader.shutdown(wait=True)

    def test_load_cleanup_after_failure(self):
        """Test that pending loads are cleaned up even after failure."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.side_effect = RuntimeError("Load failed")

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)

        # when
        future = loader.add_model_async(manager, "test/1", "key")

        with pytest.raises(RuntimeError):
            future.result(timeout=5.0)

        # then - should still be cleaned up
        stats = loader.get_stats()
        assert stats["pending_loads"] == 0

        loader.shutdown(wait=True)


class TestEnsureModelLoaded:
    """Test the ensure_model_loaded helper function."""

    def test_already_loaded_returns_immediately(self):
        """Test that ensure_model_loaded returns fast if model is loaded."""
        # given
        from inference.core.workflows.execution_engine.v1.executor.models import (
            ensure_model_loaded,
        )

        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        manager.add_model(model_id="test/1", api_key="key")

        # when
        start = time.time()
        ensure_model_loaded(manager, "test/1", "key")
        elapsed = time.time() - start

        # then - should be very fast (no actual load)
        assert elapsed < 0.1

    def test_cold_load_waits_for_completion(self):
        """Test that ensure_model_loaded waits for async load to complete."""
        # given
        from inference.core.workflows.execution_engine.v1.executor.models import (
            ensure_model_loaded,
        )

        model_registry = MagicMock()

        def slow_model(*args, **kwargs):
            time.sleep(0.2)
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = slow_model

        manager = ModelManager(model_registry=model_registry)

        # when
        ensure_model_loaded(manager, "test/1", "key", timeout=5.0)

        # then
        assert "test/1" in manager

    def test_timeout_raises_error(self):
        """Test that ensure_model_loaded raises TimeoutError on timeout."""
        # given
        from inference.core.workflows.execution_engine.v1.executor.models import (
            ensure_model_loaded,
        )

        model_registry = MagicMock()

        def very_slow_model(*args, **kwargs):
            time.sleep(10.0)
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = very_slow_model

        manager = ModelManager(model_registry=model_registry)

        # when/then
        with pytest.raises(TimeoutError, match="failed to load within"):
            ensure_model_loaded(manager, "test/1", "key", timeout=0.1)


class TestCrossManagerRegression:
    """Regression tests for cross-manager scenarios (per @voropaevv feedback)."""

    def test_cross_manager_loads_register_in_each_manager(self):
        """
        Test that two different managers requesting the same model both get it registered.

        This addresses the coalescing bug where only the first manager got the model.
        Now with per-manager coalescing keys, each manager loads independently.
        """
        # given
        model_registry = MagicMock()

        load_count = {"count": 0}
        load_lock = threading.Lock()

        def counted_model(*args, **kwargs):
            with load_lock:
                load_count["count"] += 1
            time.sleep(0.2)  # Simulate load time
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = counted_model

        manager1 = ModelManager(model_registry=model_registry)
        manager2 = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=4)

        # when - both managers request the same model concurrently
        def load_in_manager1():
            future = loader.add_model_async(manager1, "test/1", "key")
            if future:
                future.result(timeout=5.0)

        def load_in_manager2():
            future = loader.add_model_async(manager2, "test/1", "key")
            if future:
                future.result(timeout=5.0)

        t1 = threading.Thread(target=load_in_manager1)
        t2 = threading.Thread(target=load_in_manager2)

        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        # then - both managers should have the model
        assert "test/1" in manager1, "Manager 1 should have the model"
        assert "test/1" in manager2, "Manager 2 should have the model"

        # And both should have triggered separate loads (no cross-manager coalescing)
        assert load_count["count"] == 2, f"Expected 2 loads, got {load_count['count']}"

        loader.shutdown(wait=True)

    def test_saturated_loader_pool_still_completes(self):
        """
        Test that when the loader pool is saturated, requests still complete.

        This verifies that queue admission works correctly under load.
        """
        # given
        model_registry = MagicMock()

        load_times = {}
        load_lock = threading.Lock()

        def slow_model(model_id, *args, **kwargs):
            with load_lock:
                load_times[model_id] = time.time()
            time.sleep(0.3)  # Simulate slow load
            return MockModel(model_id, *args, **kwargs)

        model_registry.get_model.return_value = slow_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=2)  # Small pool to force saturation

        # when - submit 6 loads (3x the pool size)
        futures = []
        for i in range(6):
            future = loader.add_model_async(manager, f"model-{i}/1", "key")
            assert future is not None, f"Load {i} should return a Future"
            futures.append(future)

        # Wait for all to complete
        for i, future in enumerate(futures):
            future.result(timeout=10.0)
            assert f"model-{i}/1" in manager, f"Model {i} should be loaded"

        # then - all 6 models should be loaded
        assert len([k for k in manager._models.keys()]) == 6

        loader.shutdown(wait=True)

    def test_workflow_worker_blocking_behavior(self):
        """
        Test demonstrating that workflow workers ARE still blocked (per @voropaevv).

        This documents the Phase 1 limitation: calling thread blocks on future.result().
        """
        from concurrent.futures import ThreadPoolExecutor

        # given
        model_registry = MagicMock()

        block_started = threading.Event()
        block_released = threading.Event()

        def blocking_model(*args, **kwargs):
            block_started.set()
            block_released.wait(timeout=10.0)  # Block until released
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = blocking_model

        manager = ModelManager(model_registry=model_registry)
        loader = AsyncModelLoader(max_workers=4)

        # Simulate workflow thread pool
        workflow_pool = ThreadPoolExecutor(max_workers=2)
        worker_blocked = {"blocked": False}

        def workflow_step_with_load():
            """Simulates a workflow step that needs to load a model."""
            future = loader.add_model_async(manager, "test/1", "key")
            if future:
                worker_blocked["blocked"] = True
                future.result(timeout=10.0)  # THIS BLOCKS THE WORKFLOW WORKER
                worker_blocked["blocked"] = False

        # when - submit load from workflow pool
        workflow_future = workflow_pool.submit(workflow_step_with_load)

        # Wait for model load to start
        block_started.wait(timeout=5.0)

        # then - the workflow worker should be blocked
        time.sleep(0.1)  # Give it a moment to hit the blocking call
        assert worker_blocked["blocked"] == True, "Worker should be blocked waiting on Future"

        # Release the block and verify completion
        block_released.set()
        workflow_future.result(timeout=5.0)

        assert "test/1" in manager
        assert worker_blocked["blocked"] == False

        loader.shutdown(wait=True)
        workflow_pool.shutdown(wait=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
