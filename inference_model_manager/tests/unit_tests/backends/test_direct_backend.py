from types import SimpleNamespace

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


class TestRfDetrResolutionCapPassThrough:
    _FROM_PRETRAINED = (
        "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"
    )

    def _load(self, monkeypatch, cap, **load_kwargs):
        from unittest.mock import patch

        from inference_model_manager import configuration as cfg
        from inference_model_manager.model_manager import ModelManager

        monkeypatch.setattr(cfg, "RFDETR_ONNX_MAX_RESOLUTION", cap)
        mm = ModelManager()
        try:
            with patch(self._FROM_PRETRAINED) as fp:
                fp.return_value = SimpleNamespace()
                mm.load("m-cap", api_key="k", warmup_iters=0, **load_kwargs)
            return fp.call_args.kwargs
        finally:
            mm.shutdown()

    def test_manager_passes_the_configured_cap_to_from_pretrained(self, monkeypatch):
        kwargs = self._load(monkeypatch, cap=1600)

        assert kwargs["rf_detr_max_input_resolution"] == 1600

    def test_disabled_cap_is_passed_as_none(self, monkeypatch):
        kwargs = self._load(monkeypatch, cap=None)

        assert kwargs["rf_detr_max_input_resolution"] is None

    def test_explicit_load_kwarg_wins_over_the_configured_cap(self, monkeypatch):
        kwargs = self._load(monkeypatch, cap=1600, rf_detr_max_input_resolution=800)

        assert kwargs["rf_detr_max_input_resolution"] == 800

    def test_cap_and_backend_exclusions_are_forwarded_to_dependency_models(
        self, monkeypatch
    ):
        from inference_models.models.auto_loaders.core import (
            DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS,
        )

        kwargs = self._load(monkeypatch, cap=1600)

        assert kwargs["forwarded_kwargs"] == [
            *DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS,
            "rf_detr_max_input_resolution",
            "disabled_backends",
        ]

    def test_explicit_none_forwarded_kwargs_is_treated_as_omitted(self, monkeypatch):
        from inference_models.models.auto_loaders.core import (
            DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS,
        )

        kwargs = self._load(monkeypatch, cap=1600, forwarded_kwargs=None)

        assert kwargs["forwarded_kwargs"] == [
            *DEFAULT_KWARGS_PARAMS_TO_BE_FORWARDED_TO_DEPENDENT_MODELS,
            "rf_detr_max_input_resolution",
            "disabled_backends",
        ]

    def test_caller_forwarded_kwargs_get_the_manager_names_appended_once(
        self, monkeypatch
    ):
        kwargs = self._load(
            monkeypatch, cap=1600, forwarded_kwargs=["device", "disabled_backends"]
        )

        assert kwargs["forwarded_kwargs"] == [
            "device",
            "disabled_backends",
            "rf_detr_max_input_resolution",
        ]


class TestDisabledBackendsPassThrough:
    _FROM_PRETRAINED = (
        "inference_models.models.auto_loaders.core.AutoModel.from_pretrained"
    )

    def _load(self, monkeypatch, disabled, **load_kwargs):
        from unittest.mock import patch

        from inference_model_manager import configuration as cfg
        from inference_model_manager.pipelines import load_model

        monkeypatch.setattr(cfg, "DISABLED_INFERENCE_MODELS_BACKENDS", disabled)
        with patch(self._FROM_PRETRAINED) as fp:
            load_model("m-backends", "k", **load_kwargs)
        return fp.call_args.kwargs

    def test_no_disabled_backends_leaves_negotiation_to_the_library(self, monkeypatch):
        kwargs = self._load(monkeypatch, disabled=set())

        assert "disabled_backends" not in kwargs
        assert "backend" not in kwargs

    def test_disabled_backends_are_passed_explicitly(self, monkeypatch):
        kwargs = self._load(monkeypatch, disabled={"trt", "onnx"})

        assert kwargs["disabled_backends"] == ["onnx", "trt"]
        assert "backend" not in kwargs

    def test_explicit_disabled_backends_win(self, monkeypatch):
        kwargs = self._load(
            monkeypatch, disabled={"trt", "onnx"}, disabled_backends=["coreml"]
        )

        assert kwargs["disabled_backends"] == ["coreml"]

    def test_explicit_backend_is_passed_through(self, monkeypatch):
        kwargs = self._load(monkeypatch, disabled={"trt", "onnx"}, backend="trt")

        assert kwargs["backend"] == "trt"
        assert kwargs["disabled_backends"] == ["onnx", "trt"]


class TestDisabledBackendsConfiguration:
    def _reload(self, monkeypatch, value):
        import importlib

        from inference_model_manager import configuration as cfg

        monkeypatch.setenv("DISABLED_INFERENCE_MODELS_BACKENDS", value)
        try:
            return importlib.reload(cfg).DISABLED_INFERENCE_MODELS_BACKENDS
        finally:
            monkeypatch.delenv("DISABLED_INFERENCE_MODELS_BACKENDS")
            importlib.reload(cfg)

    def test_unset_means_no_restriction(self, monkeypatch):
        assert self._reload(monkeypatch, "") == set()

    def test_disabled_backends_are_parsed_and_stripped(self, monkeypatch):
        assert self._reload(monkeypatch, "trt, onnx") == {"trt", "onnx"}

    def test_unknown_backend_is_rejected(self, monkeypatch):
        with pytest.raises(ValueError, match="DISABLED_INFERENCE_MODELS_BACKENDS"):
            self._reload(monkeypatch, "mediapipe")


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


def _load_direct_backend(monkeypatch, memory_samples):
    import inference_model_manager.backends.direct as direct_mod
    import inference_model_manager.pipelines as pipelines_mod

    model = SimpleNamespace(
        _inference_config=SimpleNamespace(
            network_input=SimpleNamespace(
                dynamic_spatial_size_supported=False,
                training_input_size=SimpleNamespace(height=480, width=640),
            ),
        )
    )
    samples = iter(memory_samples)
    monkeypatch.setattr(direct_mod, "_device_memory_in_use", lambda: next(samples))
    monkeypatch.setattr(
        direct_mod,
        "time",
        SimpleNamespace(monotonic=lambda: 1234.5),
    )
    monkeypatch.setattr(pipelines_mod, "load_model", lambda *args, **kwargs: model)

    backend = DirectBackend("m", "k", device="cpu")

    return backend


class TestModelDescriptionInStats:
    def test_stats_report_input_size_and_vram_delta(self, monkeypatch):
        backend = _load_direct_backend(monkeypatch, [100, 350])

        stats = backend.stats()

        assert stats["input_height"] == 480
        assert stats["input_width"] == 640
        assert stats["vram_bytes"] == 250
        assert stats["loaded_monotonic"] == 1234.5

    def test_stats_report_no_vram_when_memory_is_unavailable(self, monkeypatch):
        backend = _load_direct_backend(monkeypatch, [None, None])

        assert backend.stats()["vram_bytes"] is None

    def test_memory_sampler_reports_none_without_cuda(self, monkeypatch):
        import torch

        from inference_model_manager.backends.direct import _device_memory_in_use

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        assert _device_memory_in_use() is None

    def test_memory_sampler_reports_device_memory_in_use(self, monkeypatch):
        import torch

        from inference_model_manager.backends.direct import _device_memory_in_use

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (300, 1000))

        assert _device_memory_in_use() == 700

    def test_memory_sampler_reports_none_when_query_raises(self, monkeypatch):
        import torch

        from inference_model_manager.backends.direct import _device_memory_in_use

        def _raise():
            raise RuntimeError("no device")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "mem_get_info", _raise)

        assert _device_memory_in_use() is None


class TestRequestedDeviceReporting:
    def test_explicit_device_is_reported(self):
        backend = DirectBackend.__new__(DirectBackend)
        backend._device_str = "cuda:1"

        assert backend._detect_device() == "cuda:1"

    def test_absent_device_falls_back_to_the_library_default(self, monkeypatch):
        from inference_models import configuration

        monkeypatch.setattr(configuration, "DEFAULT_DEVICE_STR", "cuda")
        backend = DirectBackend.__new__(DirectBackend)
        backend._device_str = None

        assert backend._detect_device() == "cuda"

    def test_model_attributes_do_not_decide_the_reported_device(self, monkeypatch):
        from types import SimpleNamespace

        from inference_models import configuration

        monkeypatch.setattr(configuration, "DEFAULT_DEVICE_STR", "cuda")
        backend = DirectBackend.__new__(DirectBackend)
        backend._device_str = None
        backend._model = SimpleNamespace(
            parameters=lambda: [SimpleNamespace(device="cpu")],
            buffers=lambda: [SimpleNamespace(device="cpu")],
        )

        assert backend._detect_device() == "cuda"


class TestBackendStateVocabulary:
    def test_members_are_interchangeable_with_the_plain_strings(self):
        import json

        from inference_model_manager.backends.base import BackendState

        assert BackendState.LOADED == "loaded"
        assert json.dumps({"state": BackendState.LOADED}) == '{"state": "loaded"}'
        assert {"loaded": 1}[BackendState.LOADED] == 1
        assert BackendState("loaded") is BackendState.LOADED

    def test_members_render_as_their_value(self):
        from inference_model_manager.backends.base import BackendState

        assert f"{BackendState.LOADED}" == "loaded"
        assert str(BackendState.LOADED) == "loaded"
        assert "%s" % BackendState.LOADED == "loaded"

    def test_a_backend_reporting_a_plain_string_still_counts_as_loaded(self):
        from inference_model_manager.model_manager import ModelManager

        mm = ModelManager()
        try:
            mm._backends["legacy-backend"] = SimpleNamespace(state="loaded")
            assert mm.loaded_models == ["legacy-backend"]
        finally:
            mm._backends.clear()
            mm.shutdown()
