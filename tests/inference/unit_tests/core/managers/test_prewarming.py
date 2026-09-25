"""
Tests for model pre-warming infrastructure (#2448).

Tests the pre-warming API that eliminates cold starts in production.
"""
import threading
import time
from unittest.mock import MagicMock

import pytest

from inference.core.managers.base import ModelManager
from inference.core.managers.prewarming import (
    ModelPrewarmConfig,
    ModelPrewarmingManager,
    PrewarmStatus,
)


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_id: str, api_key: str, **kwargs):
        self.model_id = model_id
        self.api_key = api_key
        self.task_type = "object-detection"

    def clear_cache(self, delete_from_disk: bool = True):
        pass


class TestModelPrewarming:
    """Test model pre-warming functionality."""

    def test_prewarm_single_model_success(self):
        """Test successfully pre-warming a single model."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [
            ModelPrewarmConfig("test/1", "key", pin=True),
        ]

        prewarm_mgr = ModelPrewarmingManager(manager, prewarm_configs)

        # when
        success = prewarm_mgr.warmup(timeout=10.0)

        # then
        assert success is True
        assert prewarm_mgr.is_ready() is True
        assert "test/1" in manager

        metrics = prewarm_mgr.get_metrics()
        assert metrics["loaded"] == 1
        assert metrics["failed"] == 0
        assert metrics["ready"] is True

    def test_prewarm_multiple_models_parallel(self):
        """Test pre-warming multiple models in parallel."""
        # given
        model_registry = MagicMock()

        load_times = {}
        load_lock = threading.Lock()

        def timed_model(model_id, *args, **kwargs):
            with load_lock:
                load_times[model_id] = time.time()
            time.sleep(0.2)  # Simulate load time
            return MockModel(model_id, *args, **kwargs)

        model_registry.get_model.return_value = timed_model

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [
            ModelPrewarmConfig(f"model-{i}/1", "key", pin=True)
            for i in range(4)
        ]

        prewarm_mgr = ModelPrewarmingManager(
            manager, prewarm_configs, max_parallel_loads=4
        )

        # when
        start = time.time()
        success = prewarm_mgr.warmup(timeout=10.0)
        elapsed = time.time() - start

        # then
        assert success is True
        assert len(manager._models) == 4

        # Verify parallel loading (should take ~0.2s, not 0.8s)
        assert elapsed < 0.5, f"Parallel load took too long: {elapsed}s"

        # Verify all models loaded
        for i in range(4):
            assert f"model-{i}/1" in manager

    def test_prewarm_with_retry_on_failure(self):
        """Test retry logic when model load fails initially."""
        # given
        model_registry = MagicMock()

        attempt_count = {"count": 0}

        def failing_then_success(*args, **kwargs):
            attempt_count["count"] += 1
            if attempt_count["count"] < 2:
                raise RuntimeError("Simulated load failure")
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = failing_then_success

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [ModelPrewarmConfig("test/1", "key")]

        prewarm_mgr = ModelPrewarmingManager(
            manager,
            prewarm_configs,
            retry_count=2,
            retry_delay_seconds=0.1,
        )

        # when
        success = prewarm_mgr.warmup(timeout=10.0)

        # then
        assert success is True  # Should succeed after retry
        assert attempt_count["count"] == 2  # Failed once, succeeded on retry
        assert "test/1" in manager

    def test_prewarm_failure_affects_readiness(self):
        """Test that failed pre-warm blocks readiness."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.side_effect = RuntimeError("Load failed")

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [
            ModelPrewarmConfig("test/1", "key", required_for_readiness=True)
        ]

        prewarm_mgr = ModelPrewarmingManager(
            manager, prewarm_configs, retry_count=0
        )

        # when
        success = prewarm_mgr.warmup(timeout=10.0)

        # then
        assert success is False
        assert prewarm_mgr.is_ready() is False

        metrics = prewarm_mgr.get_metrics()
        assert metrics["loaded"] == 0
        assert metrics["failed"] == 1
        assert metrics["ready"] is False

    def test_prewarm_with_pinning(self):
        """Test that pre-warmed models are pinned."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        # Use a manager with pin_model support
        base_manager = ModelManager(model_registry=model_registry)

        # Mock the pin_model method
        pinned = []

        def mock_pin(model_id):
            pinned.append(model_id)

        base_manager.pin_model = mock_pin

        prewarm_configs = [
            ModelPrewarmConfig("test/1", "key", pin=True),
            ModelPrewarmConfig("test/2", "key", pin=False),
        ]

        prewarm_mgr = ModelPrewarmingManager(base_manager, prewarm_configs)

        # when
        success = prewarm_mgr.warmup(timeout=10.0)

        # then
        assert success is True
        assert "test/1" in pinned  # Should be pinned
        assert "test/2" not in pinned  # Should not be pinned

        metrics = prewarm_mgr.get_metrics()
        results = {r["model_id"]: r for r in metrics["results"]}
        assert results["test/1"]["pinned"] is True
        assert results["test/2"]["pinned"] is False

    def test_prewarm_timeout(self):
        """Test pre-warming timeout behavior."""
        # given
        model_registry = MagicMock()

        def very_slow_model(*args, **kwargs):
            time.sleep(10.0)  # Takes too long
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = very_slow_model

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [ModelPrewarmConfig("test/1", "key")]

        prewarm_mgr = ModelPrewarmingManager(manager, prewarm_configs)

        # when
        start = time.time()
        success = prewarm_mgr.warmup(timeout=0.5)  # Short timeout
        elapsed = time.time() - start

        # then
        assert elapsed < 1.0, "Should timeout quickly"
        # Note: success may vary depending on timing

    def test_prewarm_metrics_collection(self):
        """Test detailed metrics collection."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [
            ModelPrewarmConfig(f"model-{i}/1", "key", pin=(i % 2 == 0))
            for i in range(5)
        ]

        prewarm_mgr = ModelPrewarmingManager(manager, prewarm_configs)

        # when
        prewarm_mgr.warmup(timeout=10.0)
        metrics = prewarm_mgr.get_metrics()

        # then
        assert metrics["total_models"] == 5
        assert metrics["loaded"] == 5
        assert metrics["failed"] == 0
        assert metrics["ready"] is True
        assert metrics["total_load_time_seconds"] > 0
        assert len(metrics["results"]) == 5

        # Check individual results
        for result in metrics["results"]:
            assert "model_id" in result
            assert "success" in result
            assert "load_time_seconds" in result
            assert "pinned" in result


class TestPrewarmingStatus:
    """Test pre-warming status tracking."""

    def test_status_progression(self):
        """Test that status progresses correctly."""
        # given
        model_registry = MagicMock()
        model_registry.get_model.return_value = MockModel

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [ModelPrewarmConfig("test/1", "key")]

        prewarm_mgr = ModelPrewarmingManager(manager, prewarm_configs)

        # when/then
        assert prewarm_mgr._status == PrewarmStatus.NOT_STARTED
        assert prewarm_mgr.is_ready() is False

        prewarm_mgr.warmup(timeout=10.0)

        assert prewarm_mgr._status == PrewarmStatus.COMPLETED
        assert prewarm_mgr.is_ready() is True

    def test_warmup_called_twice_is_noop(self):
        """Test that calling warmup() twice doesn't reload."""
        # given
        model_registry = MagicMock()
        load_count = {"count": 0}

        def counted_model(*args, **kwargs):
            load_count["count"] += 1
            return MockModel(*args, **kwargs)

        model_registry.get_model.return_value = counted_model

        manager = ModelManager(model_registry=model_registry)
        prewarm_configs = [ModelPrewarmConfig("test/1", "key")]

        prewarm_mgr = ModelPrewarmingManager(manager, prewarm_configs)

        # when
        prewarm_mgr.warmup(timeout=10.0)
        first_count = load_count["count"]

        prewarm_mgr.warmup(timeout=10.0)  # Second call
        second_count = load_count["count"]

        # then
        assert first_count == 1
        assert second_count == 1  # Should not reload


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
