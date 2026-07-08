"""Worker error text -> exception type mapping in MMPClient."""

from __future__ import annotations

import pytest

from inference_server.proxies.mmp_client import _raise_worker_error


def test_input_error_prefix_maps_to_value_error():
    with pytest.raises(ValueError) as exc_info:
        _raise_worker_error("INPUT_ERROR: no embeddings were found in the cache")
    assert "no embeddings were found in the cache" in str(exc_info.value)
    assert "INPUT_ERROR" not in str(exc_info.value)


def test_plain_error_maps_to_runtime_error():
    with pytest.raises(RuntimeError):
        _raise_worker_error("cuda out of memory")
