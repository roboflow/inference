import os

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
# Tests record usage under made-up API keys, so every payload is rejected and
# re-queued. Without this, the usage collector's atexit hook re-sends all of
# them to Roboflow when pytest exits (two 1s-timeout HTTP calls per key per
# payload), which delays exit by ~15s normally and past the CI job timeout when
# the API is slow. Nothing recorded here needs delivering.
os.environ["TELEMETRY_SHUTDOWN_FLUSH_TIMEOUT_SECONDS"] = "0"
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = (
    "[CUDAExecutionProvider,CPUExecutionProvider]"
)


@pytest.fixture
def local_video_path() -> str:
    return os.path.join(ASSETS_DIR, "example_video.mp4")
