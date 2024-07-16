import os
import tempfile
from typing import Generator

import pytest


os.environ["TELEMETRY_OPT_OUT"] = "True"


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
