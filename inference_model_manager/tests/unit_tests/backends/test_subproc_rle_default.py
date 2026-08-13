"""RLE mask-format defaulting in _process_slots.

Instance-segmentation models default mask_format="dense", which can exceed the
SHM result slot capacity. The worker must default to "rle" when the model
supports it, unless the client set mask_format explicitly. Models without
supported_mask_formats must never receive the kwarg.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import pytest

import inference_model_manager.backends.subproc as subproc
from inference_model_manager.backends.utils.shm_pool import SHMPool


class _FakeSock:
    def send_multipart(self, frames):
        pass


class _Log:
    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


class _SegModel:
    supported_mask_formats = {"dense", "rle"}


class _DetModel:
    pass


def _stats() -> dict:
    return {
        "inference_count": 0,
        "error_count": 0,
        "batch_count": 0,
        "latencies": deque(maxlen=10),
        "batch_sizes": deque(maxlen=10),
        "decode_ms": deque(maxlen=10),
        "infer_ms": deque(maxlen=10),
        "write_ms": deque(maxlen=10),
        "start_ts": time.monotonic(),
    }


@pytest.fixture()
def pool():
    p = SHMPool.create(n_slots=1, input_mb=0.5)
    yield p
    p.close()


def _write_npy_input(pool: SHMPool, slot_id: int, req_id: int) -> None:
    import io

    buf = io.BytesIO()
    np.save(buf, np.zeros((2, 2, 3), dtype=np.uint8))
    data = buf.getvalue()
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.data_memoryview(slot_id)[: len(data)] = data
    pool.mark_written(slot_id, len(data))


def _run(pool, model, params_bytes, monkeypatch) -> dict:
    seen: dict = {}

    def _capture(m, task, images, **kw):
        seen.update(kw)
        return [{"pred": True}]

    monkeypatch.setattr(subproc, "invoke_task", _capture)
    slot = pool.alloc_slot()
    _write_npy_input(pool, slot, req_id=101)
    subproc._process_slots(
        model,
        pool,
        [(slot, 101, params_bytes)],
        _FakeSock(),
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
        supports_rle=subproc._model_supports_rle(model),
    )
    return seen


def test_rle_injected_when_model_supports_it_and_client_silent(pool, monkeypatch):
    seen = _run(pool, _SegModel(), b"{}", monkeypatch)
    assert seen["mask_format"] == "rle"


def test_explicit_rle_untouched(pool, monkeypatch):
    seen = _run(pool, _SegModel(), b'{"mask_format": "rle"}', monkeypatch)
    assert seen["mask_format"] == "rle"


def test_explicit_dense_wins_over_default(pool, monkeypatch):
    seen = _run(pool, _SegModel(), b'{"mask_format": "dense"}', monkeypatch)
    assert seen["mask_format"] == "dense"


def test_model_without_supported_mask_formats_gets_no_kwarg(pool, monkeypatch):
    seen = _run(pool, _DetModel(), b"{}", monkeypatch)
    assert "mask_format" not in seen
