"""End-to-end ModelManager tests with real YOLOv8n model.

Loads model artifact from GCS, runs inference through ModelManager,
validates predictions. Covers both direct and subprocess backends,
multi-instance routing, lifecycle, and observability.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from inference_model_manager.model_manager import ModelManager


def _assert_detections(result: Any) -> None:
    """Validate result is Detections (or list of Detections) with correct shape."""
    # process() returns typed dict from registry serializer
    if isinstance(result, dict):
        assert "type" in result
        assert "roboflow-" in result["type"]
        if "xyxy" in result:
            assert result["xyxy"].ndim == 2
            assert result["xyxy"].shape[1] == 4
        return
    # submit() / subprocess may return raw Detections (not serialized)
    if isinstance(result, list):
        assert len(result) > 0
        det = result[0]
    else:
        det = result
    assert hasattr(det, "xyxy")
    assert hasattr(det, "confidence")
    assert det.xyxy.ndim == 2
    assert det.xyxy.shape[1] == 4


@pytest.mark.slow
@pytest.mark.torch_models
class TestModelManagerDirectE2E:
    """ModelManager + DirectBackend with real model artifact."""

    def test_load_and_infer_numpy(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")

        result = mm.process(yolov8n_model_path, images=dog_image_numpy)

        _assert_detections(result)

        mm.shutdown()

    def test_load_and_infer_jpeg_bytes(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        import cv2

        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")

        import imagecodecs

        _, buf = cv2.imencode(".jpg", dog_image_numpy)
        jpeg_bytes = buf.tobytes()
        # DirectBackend has no worker-side decode — must pass decoded array
        decoded = imagecodecs.imread(jpeg_bytes)
        result = mm.process(yolov8n_model_path, images=decoded)

        _assert_detections(result)

        mm.shutdown()

    def test_submit_future(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")

        future = mm.submit(yolov8n_model_path, images=dog_image_numpy)
        result = future.result(timeout=30)

        _assert_detections(result)

        mm.shutdown()

    def test_process_async(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        import asyncio

        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")

        result = asyncio.run(
            mm.process_async(yolov8n_model_path, images=dog_image_numpy)
        )

        _assert_detections(result)

        mm.shutdown()

    def test_stats_after_inference(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")
        mm.process(yolov8n_model_path, images=dog_image_numpy)
        mm.process(yolov8n_model_path, images=dog_image_numpy)

        s = mm.stats()

        assert len(s["models"]) == 1
        model_s = s["models"][0]
        assert model_s["model_id"] == yolov8n_model_path
        assert model_s["state"] == "loaded"
        assert model_s["inference_count"] == 2

        mm.shutdown()

    def test_unload_then_infer_raises(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct")
        mm.unload(yolov8n_model_path)

        with pytest.raises(KeyError, match="not loaded"):
            mm.process(yolov8n_model_path, images=dog_image_numpy)


@pytest.mark.slow
@pytest.mark.torch_models
class TestModelManagerMultiInstance:
    """Multiple instances of same model under different routing keys."""

    def test_two_instances_same_model(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(
            "yolov8n:0",
            api_key="",
            backend="direct",
            model_id_or_path=yolov8n_model_path,
        )
        mm.load(
            "yolov8n:1",
            api_key="",
            backend="direct",
            model_id_or_path=yolov8n_model_path,
        )

        assert "yolov8n:0" in mm
        assert "yolov8n:1" in mm
        assert len(mm) == 2

        r0 = mm.process("yolov8n:0", images=dog_image_numpy)
        r1 = mm.process("yolov8n:1", images=dog_image_numpy)

        _assert_detections(r0)
        _assert_detections(r1)

        mm.shutdown()

    def test_unload_one_instance_keeps_other(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(
            "yolov8n:0",
            api_key="",
            backend="direct",
            model_id_or_path=yolov8n_model_path,
        )
        mm.load(
            "yolov8n:1",
            api_key="",
            backend="direct",
            model_id_or_path=yolov8n_model_path,
        )

        mm.unload("yolov8n:0")

        assert "yolov8n:0" not in mm
        assert "yolov8n:1" in mm

        result = mm.process("yolov8n:1", images=dog_image_numpy)
        _assert_detections(result)

        mm.shutdown()


@pytest.mark.slow
@pytest.mark.torch_models
class TestModelManagerSubprocessE2E:
    """ModelManager + SubprocessBackend with real model.

    Combined into one test to avoid 7s model load per test.
    """

    def test_process_submit_stats(
        self, yolov8n_model_path: str, dog_image_numpy: np.ndarray
    ) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="subprocess")

        # process()
        result = mm.process(yolov8n_model_path, images=dog_image_numpy)
        _assert_detections(result)

        # submit() → Future
        future = mm.submit(yolov8n_model_path, images=dog_image_numpy)
        _assert_detections(future.result(timeout=30))

        # stats after 2 inferences
        s = mm.model_stats(yolov8n_model_path)
        assert s["state"] == "loaded"
        assert s["inference_count"] == 2

        mm.shutdown()


@pytest.mark.slow
@pytest.mark.torch_models
class TestModelManagerWarmup:
    """Warmup runs synthetic inference during load."""

    def test_warmup_iters(self, yolov8n_model_path: str) -> None:
        mm = ModelManager()
        mm.load(yolov8n_model_path, api_key="", backend="direct", warmup_iters=2)

        s = mm.model_stats(yolov8n_model_path)
        assert s["inference_count"] == 2

        mm.shutdown()
