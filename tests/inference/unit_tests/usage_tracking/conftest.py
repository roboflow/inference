import importlib
from unittest.mock import MagicMock

import pytest

from inference.core import roboflow_api
from inference.usage_tracking import collector

SERVICE_SECRET = "shared-secret"


@pytest.fixture
def configured_service_secret(monkeypatch):
    """Configure the shared secret that authorizes `countinference=false`.

    Two modules bind the secret at import time, and they have to agree: one
    validates an incoming secret, and one publishes it as the SDK's outbound
    forwarding authority.
    """
    monkeypatch.setattr(roboflow_api, "ROBOFLOW_SERVICE_SECRET", SERVICE_SECRET)
    monkeypatch.setattr(collector, "ROBOFLOW_SERVICE_SECRET", SERVICE_SECRET)
    return SERVICE_SECRET


@pytest.fixture
def usage_collector_with_mocked_threads():
    """
    Fixture that provides a UsageCollector instance with mocked threads.
    This prevents the actual threads from starting during tests.
    """
    import threading

    original_thread = threading.Thread
    original_event = threading.Event

    try:
        threading.Thread = MagicMock()
        threading.Event = MagicMock()

        from inference.usage_tracking import collector as collector_module

        importlib.reload(collector_module)

        usage_collector = collector_module.usage_collector
        threading.Thread = original_thread
        threading.Event = original_event

        usage_collector._usage.clear()
        if hasattr(usage_collector, "_hashed_api_keys"):
            usage_collector._hashed_api_keys.clear()
        if hasattr(usage_collector, "_resource_details"):
            usage_collector._resource_details.clear()

        yield usage_collector

    finally:
        # Mocked across the teardown reload too: reload defines a new class
        # object, so `UsageCollector.__new__`'s `_instance` guard resets and
        # `__init__` really does start both daemon threads again. Nothing
        # terminates the instance being discarded, so reloading with the real
        # `Thread` leaks a live usage sender for the rest of the process.
        threading.Thread = MagicMock()
        threading.Event = MagicMock()
        try:
            importlib.reload(collector_module)
        finally:
            threading.Thread = original_thread
            threading.Event = original_event
