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
os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = (
    "[CUDAExecutionProvider,CPUExecutionProvider]"
)


@pytest.fixture
def local_video_path() -> str:
    return os.path.join(ASSETS_DIR, "example_video.mp4")
