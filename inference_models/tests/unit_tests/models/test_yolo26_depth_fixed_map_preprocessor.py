import cv2
import numpy as np
import pytest

from inference_models.errors import ModelRuntimeError
from inference_models.models.yolo26.optimization.preprocessors import (
    _FIXED_MAP_RESIZED_SIZE,
    _FIXED_MAP_SOURCE_SIZE,
    _Exact5xFixedMapRemapper,
)


def test_fixed_map_remap_is_exact_and_reuses_immutable_coordinates() -> None:
    rng = np.random.default_rng(20260824)
    source = rng.integers(
        0,
        256,
        size=(*_FIXED_MAP_SOURCE_SIZE, 3),
        dtype=np.uint8,
    )
    expected = cv2.resize(
        source,
        (_FIXED_MAP_RESIZED_SIZE[1], _FIXED_MAP_RESIZED_SIZE[0]),
    )
    first_destination = np.empty_like(expected)
    second_destination = np.empty_like(expected)
    remapper = _Exact5xFixedMapRemapper()

    assert remapper._coordinate_map is None
    remapper.resize(source, first_destination)
    coordinate_map = remapper._coordinate_map
    remapper.resize(source, second_destination)

    assert np.array_equal(first_destination, expected)
    assert np.array_equal(second_destination, expected)
    assert remapper._coordinate_map is coordinate_map
    assert coordinate_map is not None
    assert coordinate_map.shape == (*_FIXED_MAP_RESIZED_SIZE, 2)
    assert coordinate_map.dtype == np.int16
    assert coordinate_map.nbytes == 1_327_104
    assert not coordinate_map.flags.writeable


def test_fixed_map_remap_rejects_unprofiled_source_geometry() -> None:
    remapper = _Exact5xFixedMapRemapper()
    source = np.empty((480, 640, 3), dtype=np.uint8)
    destination = np.empty((*_FIXED_MAP_RESIZED_SIZE, 3), dtype=np.uint8)

    with pytest.raises(ModelRuntimeError, match="fixed-map source"):
        remapper.resize(source, destination)

    assert remapper._coordinate_map is None
