"""Fixtures for the moved `test_base.py` under `execution_engine/entities/`.

The server-parity `guarded_image_codec` in the root suite installs the real
server codec (SSRF-gated URL fetch + `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM`
file access). Standalone Workflows has no server, so we install a minimal
permissive local-file codec here - just enough to let the local-filesystem
branch of `WorkflowImageData` succeed on `tmp_path` fixtures. Everything else
delegates to the package's `WorkflowsLocalImageCodec`.
"""

import tempfile
from typing import Any, Generator, Tuple

import cv2
import numpy as np
import pytest


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture(scope="function")
def guarded_image_codec() -> Generator[None, None, None]:
    """Install a permissive local-file codec for the duration of a test.

    Standalone Workflows refuses local-file reads by default; the codec below
    stands in for the server codec's `ALLOW_LOADING_IMAGES_FROM_LOCAL_FILESYSTEM=True`
    behavior so tests exercising `WorkflowImageData` from a real file on disk
    can proceed.
    """
    from roboflow_workflows.prototypes.image_codec import (
        WorkflowsLocalImageCodec,
        reset_image_codec,
        set_image_codec,
    )

    class _PermissiveLocalFileCodec(WorkflowsLocalImageCodec):
        def ensure_local_file_load_allowed(self, path: str) -> None:
            return None

        def load_image(
            self, value: Any, disable_preproc_auto_orient: bool = False
        ) -> Tuple[np.ndarray, bool]:
            if isinstance(value, dict) and value.get("type") == "file":
                image = cv2.imread(value["value"])
                if image is None:
                    raise FileNotFoundError(value["value"])
                return image, True
            return super().load_image(
                value, disable_preproc_auto_orient=disable_preproc_auto_orient
            )

    reset_image_codec()
    set_image_codec(_PermissiveLocalFileCodec())
    yield
    reset_image_codec()
