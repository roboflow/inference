"""Integration tests for DirectBackend with a real YOLOv8n model.

Inference is driven through ``ModelManager.process()`` — DirectBackend no
longer exposes a public ``infer_sync`` / ``submit`` API. Lifecycle and
observability are exercised on the backend directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from inference_model_manager.backends.direct import DirectBackend
from inference_model_manager.model_manager import ModelManager


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendPipeline:
    """Full pipeline: ModelManager.process() → backend.model → typed result."""

    def test_process_single_image(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")
        try:
            result = mm.process(yolov8n_model_path, images=dog_image_numpy)
        finally:
            mm.shutdown()

        # YOLOv8n returns List[Detections] (or a serialized typed dict via
        # registry when one is registered for the model class).
        assert result is not None

    def test_process_batch(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")
        try:
            result = mm.process(
                yolov8n_model_path, images=[dog_image_numpy, dog_image_numpy]
            )
        finally:
            mm.shutdown()

        assert result is not None
        assert len(result) == 2


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendLifecycle:

    def test_state_transitions(self, yolov8n_model_path: str) -> None:
        backend = DirectBackend(yolov8n_model_path, api_key="")

        assert backend.state == "loaded"
        assert backend.is_healthy is True
        assert backend.is_accepting is True

        backend.unload()

        assert backend.state == "unhealthy"
        assert backend.is_healthy is False
        assert backend.is_accepting is False

    def test_class_names(self, yolov8n_model_path: str) -> None:
        backend = DirectBackend(yolov8n_model_path, api_key="")
        try:
            names = backend.class_names
            assert names is not None
            assert len(names) > 0
        finally:
            backend.unload()


@pytest.mark.slow
@pytest.mark.torch_models
class TestDirectBackendObservability:

    def test_stats_populated_after_inference(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")
        try:
            mm.process(yolov8n_model_path, images=dog_image_numpy)
            mm.process(yolov8n_model_path, images=dog_image_numpy)
            s = mm.model_stats(yolov8n_model_path)
        finally:
            mm.shutdown()

        assert s["model_id"] == yolov8n_model_path
        assert s["backend_type"] == "direct"
        assert s["inference_count"] == 2
        assert s["error_count"] == 0
        assert s["latency_p50_ms"] > 0
        assert s["throughput_fps"] > 0

    def test_queue_depth_zero(self, yolov8n_model_path: str) -> None:
        backend = DirectBackend(yolov8n_model_path, api_key="")
        try:
            assert backend.queue_depth == 0
        finally:
            backend.unload()
