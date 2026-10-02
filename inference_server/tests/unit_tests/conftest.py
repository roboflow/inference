import pytest

from tests.unit_tests.legacy.conftest import (  # noqa: F401
    FakeGateway,
    fake_stat,
    key_gated_stat,
    legacy_client,
)


@pytest.fixture(autouse=True)
def _active_learning_off(monkeypatch):
    monkeypatch.setattr("inference_server.configuration.ACTIVE_LEARNING_ENABLED", False)


class _InertUsageCollector:
    def start(self):
        pass

    def record_usage(self, **row):
        pass

    def flush(self):
        pass

    def stop(self, timeout=None):
        return True


@pytest.fixture(autouse=True)
def _usage_reporting_off(monkeypatch):
    monkeypatch.setattr("inference_server.app._start_usage_collector", lambda: None)
    monkeypatch.setattr(
        "inference_server.usage.collector.UsageCollector", _InertUsageCollector
    )
