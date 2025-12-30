import os
import tempfile
from typing import Generator

import pytest

TEST_MODULES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "test_modules")
)


@pytest.fixture(scope="function")
def empty_local_dir() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def example_model_package_dir() -> str:
    return os.path.join(TEST_MODULES_DIR, "example_model")
