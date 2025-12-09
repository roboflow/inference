"""Fixtures for WebRTC end-to-end tests with real inference server.

These fixtures support slow tests that validate the full WebRTC stack
with a real inference server.
"""

import multiprocessing
import threading
import time
from functools import partial

import pytest
import requests
import uvicorn


@pytest.fixture(scope="session")
def inference_server():
    """Start real inference server for end-to-end tests.

    This fixture starts the full inference server stack:
    - HTTP API (uvicorn)
    - Stream Manager (WebRTC worker)

    The server runs in separate processes and is cleaned up after tests.
    """
    # Import server components
    from inference.core.cache import cache
    from inference.core.env import MAX_ACTIVE_MODELS
    from inference.core.interfaces.http.http_api import HttpInterface
    from inference.core.interfaces.stream_manager.manager_app.app import start
    from inference.core.managers.active_learning import (
        BackgroundTaskActiveLearningManager,
    )
    from inference.core.managers.decorators.fixed_size_cache import WithFixedSizeCache
    from inference.core.registries.roboflow import RoboflowModelRegistry
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    # Setup model manager (similar to debugrun.py)
    model_registry = RoboflowModelRegistry(ROBOFLOW_MODEL_TYPES)
    model_manager = BackgroundTaskActiveLearningManager(
        model_registry=model_registry, cache=cache
    )
    model_manager = WithFixedSizeCache(model_manager, max_size=MAX_ACTIVE_MODELS)
    model_manager.init_pingback()

    # Create HTTP interface
    interface = HttpInterface(model_manager)
    app = interface.app

    # Start stream manager process (needs separate process)
    stream_manager_process = multiprocessing.Process(
        target=partial(start, expected_warmed_up_pipelines=0),
        daemon=True,
    )
    stream_manager_process.start()

    # Start HTTP server in thread (avoids pickle issues)
    config = uvicorn.Config(app, host="127.0.0.1", port=9001, log_level="error")
    server = uvicorn.Server(config)

    def run_server():
        import asyncio
        asyncio.run(server.serve())

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    server_url = "http://127.0.0.1:9001"
    max_wait = 30  # seconds
    start_time = time.time()
    server_ready = False

    while time.time() - start_time < max_wait:
        try:
            resp = requests.get(server_url, timeout=2)
            if resp.status_code in [200, 404]:  # Server responding
                server_ready = True
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(0.5)

    if not server_ready:
        stream_manager_process.terminate()
        raise TimeoutError(
            f"Inference server failed to start within {max_wait} seconds"
        )

    print(f"\n✓ Inference server ready at {server_url}")

    # Yield server URL for tests
    yield server_url

    # Teardown: terminate processes
    print("\n✓ Shutting down inference server...")

    # Shutdown HTTP server
    server.should_exit = True

    # Terminate stream manager process
    stream_manager_process.terminate()
    stream_manager_process.join(timeout=5)

    # Force kill if still alive
    if stream_manager_process.is_alive():
        stream_manager_process.kill()
