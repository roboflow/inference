import importlib
from unittest.mock import MagicMock

import pytest


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
        threading.Thread = original_thread
        threading.Event = original_event
        importlib.reload(collector_module)
