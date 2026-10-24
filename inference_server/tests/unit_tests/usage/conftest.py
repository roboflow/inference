from typing import Optional

import pytest

from inference_server import configuration
from inference_server.framework import model_stat
from inference_server.usage.collector import UsageCollector

SYSTEM_INFO = {
    "hostname": "host1",
    "ip_address_hash": "ab12c",
    "is_gpu_available": False,
}


@pytest.fixture(autouse=True)
def usage_settings(monkeypatch):
    settings = {
        "LEGACY_OFFLINE_MODE": False,
        "LAMBDA": False,
        "GCP_SERVERLESS": False,
        "REDIS_HOST": None,
        "TELEMETRY_USE_PERSISTENT_QUEUE": False,
        "TELEMETRY_QUEUE_SIZE": 10,
        "TELEMETRY_FLUSH_INTERVAL": 10,
        "TELEMETRY_API_USAGE_ENDPOINT_URL": "https://api.example.com/usage/inference",
        "DEDICATED_DEPLOYMENT_ID": None,
        "ROBOFLOW_INTERNAL_SERVICE_SECRET": None,
        "ROBOFLOW_INTERNAL_SERVICE_NAME": None,
        "ROBOFLOW_API_EXTRA_HEADERS": None,
        "SERVER_VERSION": "9.9.9",
    }
    for name, value in settings.items():
        monkeypatch.setattr(configuration, name, value)
    monkeypatch.setattr(
        "inference_models.weights_providers.roboflow.SECURE_GATEWAY", None
    )


@pytest.fixture(autouse=True)
def _reset_model_stat_cache():
    model_stat._reset_cache_for_tests()
    yield
    model_stat._reset_cache_for_tests()


@pytest.fixture
def collector():
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)

    return usage_collector


class FakeUsageCollector:
    def __init__(self):
        self.rows = []
        self.error: Optional[Exception] = None

    def record_usage(self, **row):
        if self.error is not None:
            raise self.error
        self.rows.append(row)

    def flush(self):
        pass

    def stop(self, timeout=None):
        return True


@pytest.fixture
def usage_collector():
    return FakeUsageCollector()


@pytest.fixture
def usage_client(legacy_client, usage_collector, monkeypatch):
    monkeypatch.setattr(
        "inference_server.app._start_usage_collector", lambda: usage_collector
    )

    return legacy_client
