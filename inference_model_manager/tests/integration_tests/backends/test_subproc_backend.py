"""Integration tests for SubprocessBackend with a real YOLOv8n model.

All tests use use_gpu=False to ensure they work on CPU-only CI machines.
The SHM transport path is exercised regardless of GPU availability.

Two test groups:
  1. Raw SubprocessBackend tests: create pool externally, use submit_slot()
  2. Through ModelManager: tests the real production flow

Backend startup is expensive (~10s: spawn process, import torch, load model).
Tests that don't call unload() share a single instance via module fixtures.
"""

from __future__ import annotations

import uuid
from concurrent.futures import Future

import numpy as np
import pytest
import torch

from inference_model_manager.backends.subproc import SubprocessBackend, _to_bytes
from inference_model_manager.backends.utils.shm_pool import SHMPool
from inference_model_manager.model_manager import ModelManager

_N_SLOTS = 8
_INPUT_MB = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _submit_via_pool(pool: SHMPool, backend: SubprocessBackend, raw_input) -> Future:
    """Manually alloc slot, write input, signal worker — raw backend test path."""
    input_bytes = _to_bytes(raw_input)
    req_id = uuid.uuid4().int & 0xFFFF_FFFF_FFFF_FFFF
    slot_id = pool.alloc_slot()
    pool.mark_allocated(slot_id, req_id)
    pool.data_memoryview(slot_id)[: len(input_bytes)] = input_bytes
    pool.mark_written(slot_id, len(input_bytes))

    future: Future = Future()

    def _on_done(f):
        try:
            pool.free_slot(slot_id)
        except Exception:
            pass

    future.add_done_callback(_on_done)
    backend.submit_slot(slot_id, req_id, future)
    return future


# ---------------------------------------------------------------------------
# Fixtures — raw backend (external pool)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def shared_pool():
    pool = SHMPool.create(_N_SLOTS, _INPUT_MB)
    yield pool
    pool.close()


@pytest.fixture(scope="module")
def subproc_backend(yolov8n_model_path: str, shared_pool: SHMPool):
    """Shared SubprocessBackend for non-destructive tests."""
    backend = SubprocessBackend(
        yolov8n_model_path,
        api_key="",
        shm_pool_name=shared_pool.name,
        n_slots=_N_SLOTS,
        input_mb=_INPUT_MB,
        use_gpu=False,
        use_cuda_ipc=False,
    )
    yield backend
    backend.unload()


# ---------------------------------------------------------------------------
# Raw SubprocessBackend tests (manual pool + submit_slot)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBackendPipeline:
    """Full pipeline via worker process, raw backend path."""

    def test_infer_via_submit_slot(
        self,
        subproc_backend: SubprocessBackend,
        shared_pool: SHMPool,
        dog_image_numpy: np.ndarray,
    ) -> None:
        future = _submit_via_pool(shared_pool, subproc_backend, dog_image_numpy)
        result = future.result(timeout=30)
        assert result is not None
        assert hasattr(result, "xyxy")
        assert hasattr(result, "confidence")
        assert result.xyxy.shape[1] == 4

    def test_multiple_sequential_inferences(
        self,
        subproc_backend: SubprocessBackend,
        shared_pool: SHMPool,
        dog_image_numpy: np.ndarray,
    ) -> None:
        f1 = _submit_via_pool(shared_pool, subproc_backend, dog_image_numpy)
        r1 = f1.result(timeout=30)
        f2 = _submit_via_pool(shared_pool, subproc_backend, dog_image_numpy)
        r2 = f2.result(timeout=30)
        assert torch.allclose(
            r1.confidence.cpu(),
            r2.confidence.cpu(),
            atol=0.01,
        )


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBackendObservability:

    def test_worker_is_alive(self, subproc_backend: SubprocessBackend) -> None:
        assert subproc_backend.is_healthy is True
        assert subproc_backend.is_accepting is True
        assert subproc_backend.state == "loaded"
        assert subproc_backend.stats()["worker_alive"] is True

    def test_stats_populated_after_inference(
        self,
        subproc_backend: SubprocessBackend,
        shared_pool: SHMPool,
        dog_image_numpy: np.ndarray,
    ) -> None:
        _submit_via_pool(shared_pool, subproc_backend, dog_image_numpy).result(
            timeout=30
        )
        subproc_backend.refresh_worker_stats(timeout_s=2.0)
        s = subproc_backend.stats()
        assert s["backend_type"] == "subprocess"
        assert s["transport"] == "shm_pool"
        assert s["state"] == "loaded"
        assert s["inference_count"] >= 1
        assert s["error_count"] == 0
        assert s["latency_p50_ms"] > 0
        assert s["worker_alive"] is True

    def test_class_names(self, subproc_backend: SubprocessBackend) -> None:
        names = subproc_backend.class_names
        assert names is not None
        assert len(names) > 0


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBackendLifecycle:
    """Own pool + backend per test because they call unload()."""

    def test_worker_terminates_on_unload(self, yolov8n_model_path: str) -> None:
        pool = SHMPool.create(_N_SLOTS, _INPUT_MB)
        backend = SubprocessBackend(
            yolov8n_model_path,
            api_key="",
            shm_pool_name=pool.name,
            n_slots=_N_SLOTS,
            input_mb=_INPUT_MB,
            use_gpu=False,
            use_cuda_ipc=False,
        )
        assert backend._worker.is_alive()
        backend.unload()
        backend._worker.join(timeout=5)
        assert not backend._worker.is_alive()
        assert backend.state == "unhealthy"
        assert backend.is_accepting is False
        pool.close()

    def test_submit_after_unload_raises(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        pool = SHMPool.create(_N_SLOTS, _INPUT_MB)
        backend = SubprocessBackend(
            yolov8n_model_path,
            api_key="",
            shm_pool_name=pool.name,
            n_slots=_N_SLOTS,
            input_mb=_INPUT_MB,
            use_gpu=False,
        )
        backend.unload()
        with pytest.raises(RuntimeError, match="not available"):
            backend.submit(dog_image_numpy)
        pool.close()


# ---------------------------------------------------------------------------
# Through ModelManager (production flow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def subproc_manager(yolov8n_model_path: str):
    mm = ModelManager(n_slots=_N_SLOTS, input_mb=_INPUT_MB)
    mm.load(
        "test-model",
        api_key="",
        model_id_or_path=yolov8n_model_path,
        backend="subprocess",
        use_gpu=False,
    )
    yield mm
    mm.shutdown()


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocViaModelManager:

    def test_process(
        self, subproc_manager: ModelManager, dog_image_numpy: np.ndarray
    ) -> None:
        result = subproc_manager.process("test-model", images=dog_image_numpy)
        assert result is not None
        # process() returns typed dict from registry serializer
        if isinstance(result, dict):
            assert "type" in result
            assert "xyxy" in result
        else:
            assert hasattr(result, "xyxy")

    def test_submit_returns_future(
        self, subproc_manager: ModelManager, dog_image_numpy: np.ndarray
    ) -> None:
        f = subproc_manager.submit("test-model", raw_input=dog_image_numpy)
        result = f.result(timeout=30)
        assert result is not None
        # submit() returns raw pickle — not serialized through registry
        if isinstance(result, dict):
            assert "xyxy" in result
        else:
            assert hasattr(result, "xyxy")


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBatchingViaModelManager:

    def test_batched_submit(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager(n_slots=_N_SLOTS, input_mb=_INPUT_MB)
        mm.load(
            "m",
            api_key="",
            model_id_or_path=yolov8n_model_path,
            backend="subprocess",
            use_gpu=False,
            batch_max_size=4,
            batch_max_delay_ms=100,
        )
        f1 = mm.submit("m", raw_input=dog_image_numpy)
        f2 = mm.submit("m", raw_input=dog_image_numpy)
        r1 = f1.result(timeout=30)
        r2 = f2.result(timeout=30)
        assert r1 is not None
        assert r2 is not None
        assert hasattr(r1, "xyxy")
        assert hasattr(r2, "xyxy")
        mm.shutdown()
