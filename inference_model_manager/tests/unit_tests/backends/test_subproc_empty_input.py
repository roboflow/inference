"""Params-only (empty-input) slot handling in _process_slots.

A slot written with input_size 0 is a params-only request: the worker calls
the model with images=None and lets the model resolve inputs from params
(e.g. cached embeddings referenced by image_hashes). Empty slots are never
grouped into a shared sub-batch.
"""

from __future__ import annotations

import pickle
import time
from collections import deque

import pytest

import inference_model_manager.backends.subproc as subproc
from inference_model_manager.backends.utils.shm_pool import SHMPool


class _FakeSock:
    def __init__(self):
        self.frames = []

    def send_multipart(self, frames):
        self.frames.append(frames)


class _Log:
    def error(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass

    def exception(self, *a, **k):
        pass


class _Model:
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
    p = SHMPool.create(n_slots=2, input_mb=0.5)
    yield p
    p.close()


def _write_empty_input(pool: SHMPool, slot_id: int, req_id: int) -> None:
    pool.mark_allocated(slot_id, request_id=req_id)
    pool.mark_written(slot_id, 0)


def _run(pool, batch, capture_calls, monkeypatch, sock=None):
    def _capture(m, task=None, images="MISSING", **kw):
        capture_calls.append((task, images, kw))
        return [{"ok": True}]

    monkeypatch.setattr(subproc, "invoke_task", _capture)
    subproc._process_slots(
        _Model(),
        pool,
        batch,
        sock or _FakeSock(),
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
        supports_rle=False,
    )


def test_empty_slot_invokes_model_with_images_none(pool, monkeypatch):
    calls = []
    slot = pool.alloc_slot()
    _write_empty_input(pool, slot, req_id=7)
    _run(
        pool,
        [(slot, 7, b'{"task": "segment_with_visual_prompts", "image_hashes": ["h1"]}')],
        calls,
        monkeypatch,
    )
    assert len(calls) == 1
    task, images, kw = calls[0]
    assert task == "segment_with_visual_prompts"
    assert images is None
    assert kw["image_hashes"] == ["h1"]


def test_empty_slot_writes_result_not_error(pool, monkeypatch):
    calls = []
    sock = _FakeSock()
    slot = pool.alloc_slot()
    _write_empty_input(pool, slot, req_id=8)
    _run(pool, [(slot, 8, b'{"image_hashes": ["h1"]}')], calls, monkeypatch, sock=sock)
    hdr = pool.read_header(slot)
    data = bytes(pool.data_memoryview(slot)[: hdr.result_size])
    assert pickle.loads(data) == {"ok": True}


def test_two_empty_slots_same_params_invoke_separately(pool, monkeypatch):
    calls = []
    s1 = pool.alloc_slot()
    s2 = pool.alloc_slot()
    _write_empty_input(pool, s1, req_id=9)
    _write_empty_input(pool, s2, req_id=10)
    params = b'{"image_hashes": ["h1"]}'
    _run(pool, [(s1, 9, params), (s2, 10, params)], calls, monkeypatch)
    assert len(calls) == 2
    assert all(images is None for _, images, _ in calls)


def test_model_input_error_gets_typed_prefix(pool, monkeypatch):
    from inference_models.errors import ModelInputError

    from inference_model_manager.backends.utils.shm_pool import INPUT_ERROR_PREFIX

    def _raise(m, task=None, images=None, **kw):
        raise ModelInputError(message="no embeddings were found in the cache")

    monkeypatch.setattr(subproc, "invoke_task", _raise)
    slot = pool.alloc_slot()
    _write_empty_input(pool, slot, req_id=11)
    subproc._process_slots(
        _Model(),
        pool,
        [(slot, 11, b'{"image_hashes": ["h-unknown"]}')],
        _FakeSock(),
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
        supports_rle=False,
    )
    hdr = pool.read_header(slot)
    text = bytes(pool.data_memoryview(slot)[: hdr.result_size]).decode()
    assert text.startswith(INPUT_ERROR_PREFIX)
    assert "no embeddings were found in the cache" in text


def test_generic_error_gets_no_prefix(pool, monkeypatch):
    from inference_model_manager.backends.utils.shm_pool import INPUT_ERROR_PREFIX

    def _raise(m, task=None, images=None, **kw):
        raise RuntimeError("cuda out of memory")

    monkeypatch.setattr(subproc, "invoke_task", _raise)
    slot = pool.alloc_slot()
    _write_empty_input(pool, slot, req_id=12)
    subproc._process_slots(
        _Model(),
        pool,
        [(slot, 12, b"{}")],
        _FakeSock(),
        lambda mvs: [None] * len(mvs),
        _Log(),
        _stats(),
        supports_rle=False,
    )
    hdr = pool.read_header(slot)
    text = bytes(pool.data_memoryview(slot)[: hdr.result_size]).decode()
    assert not text.startswith(INPUT_ERROR_PREFIX)
