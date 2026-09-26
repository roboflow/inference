"""
Tests for eviction protection (#2448).

Tests the usage tracking and protection logic that prevents the evict→reload
cycle for models in active rotation.
"""
import time
from unittest.mock import MagicMock

import pytest

from inference.core.managers.eviction_protection import (
    EvictionProtectionManager,
    UsageStats,
)


class TestEvictionProtection:
    """Test eviction protection functionality."""

    def test_recently_used_model_is_protected(self):
        """Test that recently-used models are protected from eviction."""
        # given
        protection = EvictionProtectionManager(protection_window_seconds=5.0)
        protection.record_usage("test/1")

        # when
        is_protected = protection.is_protected("test/1")

        # then
        assert is_protected is True

    def test_old_unused_model_not_protected(self):
        """Test that old unused models are not protected."""
        # given
        protection = EvictionProtectionManager(protection_window_seconds=0.1)
        protection.record_usage("test/1")

        # Wait for protection window to expire
        time.sleep(0.15)

        # when
        is_protected = protection.is_protected("test/1")

        # then
        assert is_protected is False

    def test_never_used_model_not_protected(self):
        """Test that models never used are not protected."""
        # given
        protection = EvictionProtectionManager()

        # when
        is_protected = protection.is_protected("test/1")

        # then
        assert is_protected is False

    def test_high_frequency_model_protected(self):
        """Test that high-frequency models are protected even if not recent."""
        # given
        protection = EvictionProtectionManager(
            protection_window_seconds=0.1,
            high_frequency_threshold=5,
        )

        # Simulate 10 uses
        for _ in range(10):
            protection.record_usage("test/1")
            time.sleep(0.01)

        # Wait for protection window to expire
        time.sleep(0.15)

        # when
        is_protected = protection.is_protected("test/1")

        # then
        # Should still be protected due to high frequency
        assert is_protected is True

    def test_usage_tracking(self):
        """Test that usage is tracked correctly."""
        # given
        protection = EvictionProtectionManager()

        # when
        protection.record_usage("test/1")
        protection.record_usage("test/1")
        protection.record_usage("test/2")

        stats1 = protection.get_stats("test/1")
        stats2 = protection.get_stats("test/2")

        # then
        assert stats1 is not None
        assert stats1.use_count == 2
        assert stats1.model_id == "test/1"

        assert stats2 is not None
        assert stats2.use_count == 1
        assert stats2.model_id == "test/2"

    def test_cleanup_removes_inactive_models(self):
        """Test that cleanup removes stats for unloaded models."""
        # given
        protection = EvictionProtectionManager()

        protection.record_usage("test/1")
        protection.record_usage("test/2")
        protection.record_usage("test/3")

        # when
        active_models = {"test/1", "test/3"}  # test/2 is no longer active
        protection.cleanup_stats(active_models)

        # then
        assert protection.get_stats("test/1") is not None
        assert protection.get_stats("test/2") is None  # Should be cleaned up
        assert protection.get_stats("test/3") is not None

    def test_metrics_collection(self):
        """Test metrics collection."""
        # given
        protection = EvictionProtectionManager(protection_window_seconds=10.0)

        # Record some usage
        protection.record_usage("test/1")
        protection.record_usage("test/2")

        # Check protection (triggers metric counting)
        protection.is_protected("test/1")  # Should be protected
        protection.is_protected("test/3")  # Should not be protected (never used)

        # when
        metrics = protection.get_metrics()

        # then
        assert metrics["tracked_models"] == 2
        assert metrics["protection_saves"] == 1
        assert metrics["evictions_allowed"] == 1
        assert metrics["protection_rate"] == 50.0
        assert metrics["recently_used"] == 2

    def test_protection_window_configurable(self):
        """Test that protection window is configurable."""
        # given
        short_protection = EvictionProtectionManager(protection_window_seconds=0.1)
        long_protection = EvictionProtectionManager(protection_window_seconds=10.0)

        short_protection.record_usage("test/1")
        long_protection.record_usage("test/1")

        time.sleep(0.15)

        # when/then
        assert short_protection.is_protected("test/1") is False
        assert long_protection.is_protected("test/1") is True

    def test_concurrent_usage_tracking_thread_safe(self):
        """Test that usage tracking is thread-safe."""
        import threading

        # given
        protection = EvictionProtectionManager()

        def record_many_times():
            for _ in range(100):
                protection.record_usage("test/1")

        # when
        threads = [threading.Thread(target=record_many_times) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # then
        stats = protection.get_stats("test/1")
        assert stats.use_count == 1000  # 10 threads * 100 uses each

    def test_metrics_reset(self):
        """Test that metrics can be reset."""
        # given
        protection = EvictionProtectionManager()

        protection.record_usage("test/1")
        protection.is_protected("test/1")

        metrics_before = protection.get_metrics()
        assert metrics_before["protection_saves"] > 0

        # when
        protection.reset_metrics()
        metrics_after = protection.get_metrics()

        # then
        assert metrics_after["protection_saves"] == 0
        assert metrics_after["evictions_allowed"] == 0


class TestUsageStats:
    """Test UsageStats dataclass."""

    def test_usage_stats_creation(self):
        """Test creating usage stats."""
        # when
        stats = UsageStats(
            model_id="test/1",
            last_used_timestamp=time.time(),
            use_count=5,
            first_seen_timestamp=time.time() - 100,
        )

        # then
        assert stats.model_id == "test/1"
        assert stats.use_count == 5
        assert stats.last_used_timestamp > stats.first_seen_timestamp


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
