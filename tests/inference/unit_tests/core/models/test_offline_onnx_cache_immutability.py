from unittest.mock import MagicMock

import pytest


def test_offline_onnx_session_failure_never_clears_warmed_cache(
    monkeypatch,
) -> None:
    from inference.core.exceptions import ModelArtefactError
    from inference.core.models import roboflow

    model = object.__new__(roboflow.OnnxRoboflowInferenceModel)
    model.load_weights = True
    monkeypatch.setattr(
        roboflow.OnnxRoboflowInferenceModel,
        "has_model_metadata",
        False,
    )
    model.onnxruntime_execution_providers = ["CPUExecutionProvider"]
    model.get_model_artifacts = MagicMock()
    model.cache_file = MagicMock(return_value="/read-only/model.onnx")
    model.clear_cache = MagicMock()
    model.endpoint = "workspace/model/1"
    monkeypatch.setattr(
        roboflow.OnnxRoboflowInferenceModel,
        "weights_file",
        "model.onnx",
    )
    monkeypatch.setattr(roboflow, "OFFLINE_MODE", True)
    monkeypatch.setattr(
        roboflow.onnxruntime,
        "InferenceSession",
        MagicMock(side_effect=RuntimeError("bad ONNX")),
    )

    with pytest.raises(ModelArtefactError, match="Unable to load ONNX session"):
        model.initialize_model()

    model.clear_cache.assert_not_called()


def test_offline_legacy_tensorrt_tuple_cannot_reenable_file_caches() -> None:
    from inference.core.models.utils.onnx import disable_onnxruntime_trt_file_outputs

    original_options = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "/read-only/model",
        "trt_timing_cache_enable": True,
        "trt_timing_cache_path": "/read-only/model/timing",
        "trt_force_timing_cache": True,
        "trt_dump_subgraphs": True,
        "trt_dump_ep_context_model": True,
        "trt_ep_context_file_path": "/read-only/model/context.onnx",
        "trt_onnx_model_folder_path": "/read-only/model",
        "trt_fp16_enable": True,
    }

    provider_name, provider_options = disable_onnxruntime_trt_file_outputs(
        provider=("TensorrtExecutionProvider", original_options)
    )

    assert provider_name == "TensorrtExecutionProvider"
    assert provider_options["trt_engine_cache_enable"] is False
    assert provider_options["trt_timing_cache_enable"] is False
    assert provider_options["trt_force_timing_cache"] is False
    assert provider_options["trt_dump_subgraphs"] is False
    assert provider_options["trt_dump_ep_context_model"] is False
    assert not {
        "trt_engine_cache_path",
        "trt_engine_cache_prefix",
        "trt_timing_cache_path",
        "trt_ep_context_file_path",
        "trt_onnx_model_folder_path",
    }.intersection(provider_options)
    assert provider_options["trt_fp16_enable"] is True
    assert original_options["trt_engine_cache_enable"] is True
