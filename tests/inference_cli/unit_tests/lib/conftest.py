import os.path
import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def text_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), "assets", "file_with_lines.txt")


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
