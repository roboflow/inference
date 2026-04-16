"""Integration tests for SubprocessBackend with a real YOLOv8n model.

All tests use use_gpu=False to ensure they work on CPU-only CI machines.
The SHM transport path is exercised regardless of GPU availability.

Backend startup is expensive (~10s: spawn process, import torch, load model
twice). Tests that don't call unload() share a single backend instance via
class-scoped fixtures. Only lifecycle tests create their own instance.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inference_models.backends.subproc import SubprocessBackend


@pytest.fixture(scope="module")
def subproc_backend(yolov8n_model_path: str):
    """Shared SubprocessBackend for non-destructive tests."""
    backend = SubprocessBackend(
        yolov8n_model_path, api_key="",
        use_gpu=False, use_cuda_ipc=False,
    )
    yield backend
    backend.unload()


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBackendPipeline:
    """Full pipeline: pre_process → submit → post_process via worker process."""

    def test_infer_sync(
        self, subproc_backend: SubprocessBackend, dog_image_numpy: np.ndarray
    ) -> None:
        # when
        result = subproc_backend.infer_sync(dog_image_numpy)

        # then
        assert result is not None
        assert len(result) > 0
        assert hasattr(result[0], "xyxy")
        assert hasattr(result[0], "confidence")
        assert result[0].xyxy.shape[1] == 4

    def test_multiple_sequential_inferences(
        self, subproc_backend: SubprocessBackend, dog_image_numpy: np.ndarray
    ) -> None:
        # when
        r1 = subproc_backend.infer_sync(dog_image_numpy)
        r2 = subproc_backend.infer_sync(dog_image_numpy)

        # then — same input, same result
        assert len(r1) == len(r2)
        assert torch.allclose(
            r1[0].confidence.cpu(), r2[0].confidence.cpu(), atol=0.01,
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
        self, subproc_backend: SubprocessBackend, dog_image_numpy: np.ndarray
    ) -> None:
        # given — run inference to populate stats
        subproc_backend.infer_sync(dog_image_numpy)

        # when
        s = subproc_backend.stats()

        # then
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
    """These tests create their own backend because they call unload()."""

    def test_worker_terminates_on_unload(self, yolov8n_model_path: str) -> None:
        # given
        backend = SubprocessBackend(
            yolov8n_model_path, api_key="",
            use_gpu=False, use_cuda_ipc=False,
        )
        assert backend._worker.is_alive()

        # when
        backend.unload()

        # then
        assert not backend._worker.is_alive()
        assert backend.state == "unhealthy"
        assert backend.is_accepting is False

    def test_submit_after_unload_raises(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = SubprocessBackend(
            yolov8n_model_path, api_key="",
            use_gpu=False,
        )
        backend.unload()

        # when / then
        with pytest.raises(RuntimeError, match="not accepting"):
            backend.submit(dog_image_numpy)


@pytest.mark.slow
@pytest.mark.torch_models
class TestSubprocBackendBatching:
    """Batching needs its own backend (different config)."""

    def test_batched_submit(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = SubprocessBackend(
            yolov8n_model_path,
            api_key="",
            use_gpu=False,
            batch_max_size=4,
            batch_max_delay_ms=100,
        )

        # when
        f1 = backend.submit(dog_image_numpy)
        f2 = backend.submit(dog_image_numpy)
        r1 = f1.result(timeout=30)
        r2 = f2.result(timeout=30)

        # then
        assert r1 is not None
        assert r2 is not None
        assert hasattr(r1[0], "xyxy")
        assert hasattr(r2[0], "xyxy")

        backend.unload()
