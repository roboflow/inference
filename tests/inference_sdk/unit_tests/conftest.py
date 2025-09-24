import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
