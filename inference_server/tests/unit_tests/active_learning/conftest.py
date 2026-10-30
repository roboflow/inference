import numpy as np
import pytest


@pytest.fixture(scope="function")
def image_as_numpy() -> np.ndarray:
    return np.zeros((128, 128, 3), dtype=np.uint8)
