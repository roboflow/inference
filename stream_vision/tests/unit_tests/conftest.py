import logging
import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def streamvision_caplog(
    caplog: pytest.LogCaptureFixture,
) -> Generator[pytest.LogCaptureFixture, None, None]:
    """caplog attaches to the root logger, but a host (e.g. ``inference``) may
    set ``propagate = False`` on the ``streamvision`` logger, so records never
    reach it. Attach caplog's handler directly to the ``streamvision`` logger.
    """
    streamvision_logger = logging.getLogger("streamvision")
    streamvision_logger.addHandler(caplog.handler)
    try:
        yield caplog
    finally:
        streamvision_logger.removeHandler(caplog.handler)
