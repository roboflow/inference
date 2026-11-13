import itertools
import time
from typing import Any, Optional

import pytest

from inference_server import configuration
from inference_server.framework import model_stat
from inference_server.usage.collector import UsageCollector

SYSTEM_INFO = {
    "hostname": "host1",
    "ip_address_hash": "ab12c",
    "is_gpu_available": False,
}
CLOCK_STEP_S = 0.001
# Modules that time usage rows with ``time.perf_counter``: request_hook (the
# request row), observer (workflow and block rows) and bridge (model rows of
# legacy routes and workflow model steps).
ROW_CLOCK_MODULES = (
    "inference_server.legacy.bridge",
    "inference_server.usage.observer",
    "inference_server.usage.request_hook",
)


class SteppingClock:
    """``time`` stand-in whose ``perf_counter`` moves ``CLOCK_STEP_S`` per read.

    A duration measured with it counts the clock reads inside it, so it stays
    a few milliseconds however slow the machine running the test is.
    ``itertools.count`` gives every read its own tick, whichever thread reads
    it. Other attributes are the real ``time`` module's.
    """

    def __init__(self) -> None:
        self._ticks = itertools.count(1)

    def perf_counter(self) -> float:
        now = next(self._ticks) * CLOCK_STEP_S

        return now

    def __getattr__(self, name: str) -> Any:
        return getattr(time, name)


@pytest.fixture
def stepping_clock(monkeypatch):
    clock = SteppingClock()
    for module in ROW_CLOCK_MODULES:
        monkeypatch.setattr(f"{module}.time", clock)

    return clock


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
