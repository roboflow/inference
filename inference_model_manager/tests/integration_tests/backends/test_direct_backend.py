"""Integration tests for DirectBackend with a real YOLOv8n model."""

from __future__ import annotations

import numpy as np
import pytest

from inference_model_manager.backends.direct import DirectBackend


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendPipeline:
    """Full pipeline: pre_process → submit → post_process via real model."""

    def test_infer_sync(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")

        # when
        result = backend.infer_sync(dog_image_numpy)

        # then — YOLOv8n returns List[Detections]
        assert result is not None
        assert len(result) > 0
        assert hasattr(result[0], "xyxy")
        assert hasattr(result[0], "confidence")
        assert result[0].xyxy.shape[1] == 4

        backend.unload()

    def test_batch_inference(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given — two images through infer_sync (model handles batch internally)
        backend = DirectBackend(yolov8n_model_path, api_key="")

        # when
        result = backend.infer_sync([dog_image_numpy, dog_image_numpy])

        # then — should get predictions for both images
        assert result is not None
        assert len(result) == 2

        backend.unload()


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendBatching:
    """BatchCollector integration with real model."""

    def test_batched_submit(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = DirectBackend(
            yolov8n_model_path,
            api_key="",
            batch_max_size=4,
            batch_max_delay_ms=50,
        )

        # when — submit multiple items, BatchCollector groups them
        f1 = backend.submit(dog_image_numpy)
        f2 = backend.submit(dog_image_numpy)
        r1 = f1.result(timeout=10)
        r2 = f2.result(timeout=10)

        # then
        assert r1 is not None
        assert r2 is not None

        backend.unload()


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendLifecycle:

    def test_state_transitions(self, yolov8n_model_path: str) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")

        # then — loaded
        assert backend.state == "loaded"
        assert backend.is_healthy is True
        assert backend.is_accepting is True

        # when — unload
        backend.unload()

        # then — unhealthy
        assert backend.state == "unhealthy"
        assert backend.is_healthy is False
        assert backend.is_accepting is False

    def test_submit_after_unload_raises(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")
        backend.unload()

        # when / then
        with pytest.raises(RuntimeError, match="not accepting"):
            backend.submit(dog_image_numpy)

    def test_class_names(self, yolov8n_model_path: str) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")

        # then
        names = backend.class_names
        assert names is not None
        assert len(names) > 0

        backend.unload()


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendObservability:

    def test_stats_populated_after_inference(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")
        backend.infer_sync(dog_image_numpy)
        backend.infer_sync(dog_image_numpy)

        # when
        s = backend.stats()

        # then
        assert s["model_id"] == yolov8n_model_path
        assert s["backend_type"] == "direct"
        assert s["state"] == "loaded"
        assert s["inference_count"] == 2
        assert s["error_count"] == 0
        assert s["latency_p50_ms"] > 0
        assert s["throughput_fps"] > 0

        backend.unload()

    def test_queue_depth_zero_without_batching(self, yolov8n_model_path: str) -> None:
        # given
        backend = DirectBackend(yolov8n_model_path, api_key="")

        # then
        assert backend.queue_depth == 0

        backend.unload()
