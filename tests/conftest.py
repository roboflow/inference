import os
import sys
from threading import Thread
from unittest import mock

import pytest

ASSETS_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "inference",
        "unit_tests",
        "core",
        "interfaces",
        "assets",
    )
)

os.environ["TELEMETRY_USE_PERSISTENT_QUEUE"] = "False"
# Keep the process-wide usage collector from reporting usage for fake test API
# keys to the live API. Port 9 on loopback refuses connections immediately.
os.environ.setdefault("METRICS_COLLECTOR_BASE_URL", "http://127.0.0.1:9")
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = (
    "[CUDAExecutionProvider,CPUExecutionProvider]"
)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    _stop_usage_collector_threads()


def _stop_usage_collector_threads() -> None:
    """Stop the usage collector's background threads for the rest of the process.

    Once a test records usage with an API key, the process-wide collector's
    sender thread retries sending it every flush interval until the process
    exits: a ``requests.get`` for the key's plan and a ``requests.post`` with
    the usage, which the loopback URL above refuses. A retry that lands while a
    test has patched ``requests`` or its transport adds calls to that test's
    fake. No test needs the threads: tests that exercise flushing call it
    directly.

    The module is looked up, never imported, so suites that never load it are
    unaffected. Running before every test also stops a collector that an
    earlier test imported.
    """
    collector_module = sys.modules.get("inference.usage_tracking.collector")
    usage_collector = getattr(collector_module, "usage_collector", None)
    threads = (
        getattr(usage_collector, "_collector_thread", None),
        getattr(usage_collector, "_sender_thread", None),
    )
    if not any(isinstance(thread, Thread) and thread.is_alive() for thread in threads):
        return

    # A stopping thread enqueues and flushes once more. Skip both, so stopping
    # the threads sends nothing and pending usage stays where it is.
    with mock.patch.object(usage_collector, "_enqueue_usage_payload"):
        with mock.patch.object(usage_collector, "_flush_queue"):
            usage_collector._cleanup()


@pytest.fixture
def local_video_path() -> str:
    return os.path.join(ASSETS_DIR, "example_video.mp4")
