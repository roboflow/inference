import tempfile
from typing import Generator

import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def guarded_image_codec() -> Generator[None, None, None]:
    """Install the server's image codec for tests that read a real local file.

    `WorkflowImageData` asks the installed codec for permission before touching
    the filesystem. With no host codec the workflows-local default refuses -
    which is the point of the port - so a test exercising the local-file branch
    must stand in for the server, exactly as the composition roots do.
    """
    from inference.core.interfaces.workflows_image_codec import (
        install_guarded_image_codec,
    )
    from inference.core.workflows.prototypes.image_codec import reset_image_codec

    reset_image_codec()
    install_guarded_image_codec()
    yield
    reset_image_codec()
