import logging
import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def inference_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Generator[pytest.LogCaptureFixture, None, None]:
    """caplog attaches to the root logger, but the ``inference`` logger (and
    its descendants) has ``propagate = False`` (see
    ``inference/core/logger.py``), so records never reach it. Attach
    caplog's handler directly to the ``inference`` logger instead.
    """
    inference_logger = logging.getLogger("inference")
    inference_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        inference_logger.removeHandler(caplog.handler)
