import inspect
from unittest import mock

import pytest

from inference_models.models.auto_loaders import core
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    FileDownloadSpecs,
    ModelMetadata,
    ModelPackageMetadata,
)

PARAMETERS_ON_MAIN = [
    "model_id_or_path",
    "weights_provider",
    "api_key",
    "model_package_id",
    "backend",
    "batch_size",
    "quantization",
    "onnx_execution_providers",
    "device",
    "default_onnx_trt_options",
    "max_package_loading_attempts",
    "verbose",
    "model_download_file_lock_acquire_timeout",
    "allow_untrusted_packages",
    "trt_engine_host_code_allowed",
    "allow_local_code_packages",
    "verify_hash_while_download",
    "download_files_without_hash",
    "use_auto_resolution_cache",
    "auto_resolution_cache",
    "allow_direct_local_storage_loading",
    "model_access_manager",
    "nms_fusion_preferences",
    "model_type",
    "task_type",
    "allow_loading_dependency_models",
    "dependency_models_params",
    "point_model_directory",
    "forwarded_kwargs",
    "weights_provider_extra_query_params",
    "weights_provider_extra_headers",
    "content_addressed_artifact_cache",
    "required_capabilities",
]


def test_from_pretrained_keeps_positional_binding_of_every_parameter_on_main() -> None:
    parameters = inspect.signature(core.AutoModel.from_pretrained).parameters

    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]

    assert positional == PARAMETERS_ON_MAIN
    assert set(keyword_only) == {"preloaded_model_dependencies", "disabled_backends"}
    assert all(parameters[name].default is None for name in keyword_only)


def _metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="workspace/model/1",
        model_architecture="rfdetr",
        model_packages=[
            ModelPackageMetadata(
                package_id="pkg1",
                backend=BackendType.ONNX,
                package_artefacts=[
                    FileDownloadSpecs(
                        download_url="https://weights.example/weights.onnx",
                        file_handle="weights.onnx",
                        md5_hash="ff01",
                    )
                ],
            )
        ],
        task_type="object-detection",
    )


def _capture_negotiation_backend(monkeypatch, **load_kwargs):
    monkeypatch.setenv("DISABLED_INFERENCE_MODELS_BACKENDS", "onnx")
    captured = {}

    def _record(**kwargs):
        captured["backend"] = kwargs["requested_backends"]
        raise RuntimeError("stop after capturing the backend request")

    with mock.patch.object(
        core, "get_model_from_provider", return_value=_metadata()
    ), mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", return_value=None
    ), mock.patch.object(
        core, "negotiate_model_packages", side_effect=_record
    ):
        with pytest.raises(RuntimeError, match="stop after capturing"):
            core.AutoModel.from_pretrained(
                "workspace/model/1", api_key="key", **load_kwargs
            )
    return captured


def test_disabled_backends_environment_is_ignored_unless_passed_explicitly(
    monkeypatch,
) -> None:
    captured = _capture_negotiation_backend(monkeypatch)

    assert captured == {"backend": None}


def test_disabled_backends_parameter_turns_into_an_explicit_allowed_list(
    monkeypatch,
) -> None:
    captured = _capture_negotiation_backend(monkeypatch, disabled_backends={"onnx"})

    assert captured["backend"] is not None
    assert "onnx" not in captured["backend"]
    assert "torch" in captured["backend"]
    assert captured["backend"] == sorted(captured["backend"])


def test_disabled_backends_parameter_does_not_override_an_explicit_backend(
    monkeypatch,
) -> None:
    captured = _capture_negotiation_backend(
        monkeypatch, disabled_backends={"onnx"}, backend="onnx"
    )

    assert captured["backend"] == "onnx"


def _capture_forwarded_values(**load_kwargs):
    captured = {}

    def _record(**kwargs):
        captured.update(kwargs["forwarded_kwargs_values"])
        raise RuntimeError("stop after capturing forwarded values")

    with mock.patch.object(
        core, "attempt_loading_model_with_auto_load_cache", side_effect=_record
    ):
        with pytest.raises(RuntimeError, match="stop after capturing"):
            core.AutoModel.from_pretrained(
                "workspace/model/1", api_key="key", **load_kwargs
            )
    return captured


def test_disabled_backends_reach_dependency_loads_only_when_forwarded() -> None:
    not_forwarded = _capture_forwarded_values(disabled_backends=["onnx"])
    forwarded = _capture_forwarded_values(
        disabled_backends=["onnx"], forwarded_kwargs=["disabled_backends"]
    )

    assert "disabled_backends" not in not_forwarded
    assert forwarded == {"disabled_backends": ["onnx"]}


def test_forwarded_values_do_not_override_explicit_dependency_parameters() -> None:
    from inference_models.models.auto_loaders.dependency_models import (
        prepare_dependency_model_parameters,
    )

    dependency = prepare_dependency_model_parameters(
        model_parameters={"model_id_or_path": "dep/1", "backend": "onnx"}
    )
    forwarded_values = {
        "disabled_backends": ["onnx"],
        "rf_detr_max_input_resolution": 1600,
    }

    for name, value in forwarded_values.items():
        if name not in dependency.model_extra:
            dependency.model_extra[name] = value

    assert dependency.backend == "onnx"
    assert dependency.kwargs == forwarded_values


CACHE_HELPER_PARAMETERS_ON_MAIN = [
    "use_auto_resolution_cache",
    "auto_resolution_cache",
    "auto_negotiation_hash",
    "model_access_manager",
    "model_name_or_path",
    "model_init_kwargs",
    "api_key",
    "allow_loading_dependency_models",
    "forwarded_kwargs_values",
    "verbose",
    "weights_provider",
    "max_package_loading_attempts",
    "model_download_file_lock_acquire_timeout",
    "allow_untrusted_packages",
    "trt_engine_host_code_allowed",
    "allow_local_code_packages",
    "verify_hash_while_download",
    "download_files_without_hash",
    "allow_direct_local_storage_loading",
    "dependency_models_params",
    "weights_provider_extra_query_params",
    "weights_provider_extra_headers",
    "point_model_directory",
    "content_addressed_artifact_cache",
]


def test_auto_load_cache_helper_keeps_positional_binding_of_main() -> None:
    parameters = inspect.signature(
        core.attempt_loading_model_with_auto_load_cache
    ).parameters

    positional = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    keyword_only = [
        name
        for name, parameter in parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY
    ]

    assert positional == CACHE_HELPER_PARAMETERS_ON_MAIN
    assert keyword_only == ["preloaded_model_dependencies"]
    assert parameters["preloaded_model_dependencies"].default is None
