import os.path
import tempfile
from typing import Generator

import pytest

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))


@pytest.fixture
def local_video_path() -> str:
    return os.path.join(ASSETS_DIR, "example_video.mp4")


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
