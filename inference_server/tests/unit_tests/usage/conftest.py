import pytest

from inference_server import configuration
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


@pytest.fixture
def collector():
    usage_collector = UsageCollector()
    usage_collector._system_info = dict(SYSTEM_INFO)

    return usage_collector
