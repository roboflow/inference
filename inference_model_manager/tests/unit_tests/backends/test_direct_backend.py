import numpy as np
import pytest

from inference_model_manager.backends.decode import make_decoder
from inference_model_manager.backends.direct import DirectBackend


def test_encoded_rgb_is_converted_to_bgr_without_changing_raw_numpy():
    imagecodecs = pytest.importorskip("imagecodecs")
    rgb = np.array([[[255, 1, 7], [3, 5, 251]]], dtype=np.uint8)
    encoded = bytes(imagecodecs.png_encode(rgb))
    raw_bgr = rgb[..., ::-1].copy()
    backend = DirectBackend.__new__(DirectBackend)
    backend._decoder_name = "imagecodecs"
    backend._decode = make_decoder("imagecodecs", device="cpu")

    np.testing.assert_array_equal(backend._decode_input(encoded), raw_bgr)
    assert backend._decode_input(raw_bgr) is raw_bgr


class TestTorchscriptLockPassThrough:
    def test_factory_forwards_manager_lock_to_from_pretrained(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from inference_model_manager.model_manager import ModelManager

        mm = ModelManager()
        try:
            with patch(
                "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"
            ) as fp:
                fp.return_value = SimpleNamespace()
                mm.load("m-lock", api_key="k", warmup_iters=0)
            assert (
                fp.call_args.kwargs["torchscript_state_global_lock"]
                is mm.torchscript_state_global_lock
            )
        finally:
            mm.shutdown()


class TestResolvedModelInStats:
    def test_stats_report_resolved_model_when_model_exposes_it(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from inference_model_manager.model_manager import ModelManager
        from inference_models.entities import ResolvedModelMetadata

        mm = ModelManager()
        try:
            with patch(
                "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"
            ) as fp:
                fp.return_value = SimpleNamespace(
                    resolved_model=ResolvedModelMetadata(
                        model_id="coco/3",
                        model_package_id="pkg-1",
                        backend="onnx",
                        quantization="fp32",
                    )
                )
                mm.load("m-resolved", api_key="k", warmup_iters=0)
            entry = next(
                m for m in mm.stats()["models"] if m["model_id"] == "m-resolved"
            )
            assert entry["resolved_model"] == {
                "model_id": "coco/3",
                "model_package_id": "pkg-1",
                "backend": "onnx",
                "quantization": "fp32",
            }
        finally:
            mm.shutdown()

    def test_stats_report_none_without_resolved_model(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from inference_model_manager.model_manager import ModelManager

        mm = ModelManager()
        try:
            with patch(
                "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"
            ) as fp:
                fp.return_value = SimpleNamespace()
                mm.load("m-plain", api_key="k", warmup_iters=0)
            entry = next(m for m in mm.stats()["models"] if m["model_id"] == "m-plain")
            assert entry["resolved_model"] is None
        finally:
            mm.shutdown()
