"""Fixtures shared by every moved unit-test module.

Assets are package-owned. They live under `workflows/tests/assets/` so the
standalone project has zero read-time coupling to the parent repo. Add new
assets under that dir; DO NOT reach into `../tests/workflows/...`.
"""

import tempfile
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
import pytest

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"


@pytest.fixture(scope="function")
def dogs_image() -> np.ndarray:
    return cv2.imread(str(_ASSETS_DIR / "dogs.jpg"))


@pytest.fixture(scope="function")
def empty_directory() -> Generator[str, None, None]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir
