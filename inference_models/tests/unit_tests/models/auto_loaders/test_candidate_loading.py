from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch

from inference_models.errors import (
    ModelPackageAlternativesExhaustedError,
    NoModelPackagesAvailableError,
)
from inference_models.models.auto_loaders import auto_negotiation, core
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    ModelPackageMetadata,
    Quantization,
)


def _package(package_id: str = "selected") -> ModelPackageMetadata:
    package = ModelPackageMetadata(
        package_id=package_id,
        backend=BackendType.TORCH,
        package_artefacts=[],
        quantization=Quantization.FP32,
        trusted_source=True,
        dynamic_batch_size_supported=True,
    )
    return package


def _load_arguments() -> dict:
    arguments = {
        "model_id": "canonical/1",
        "requested_model_id": "model-alias",
        "model_architecture": "rfdetr",
        "task_type": "instance-segmentation",
        "model_init_kwargs": {"trt_cuda_graph_cache": True, "device": "cuda:0"},
        "model_access_manager": MagicMock(),
        "auto_resolution_cache": MagicMock(),
        "auto_negotiation_hash": "existing-cache-key",
        "api_key": None,
        "model_dependencies": [],
        "model_dependencies_instances": {},
        "model_dependencies_directories": {},
        "point_model_directory": MagicMock(),
        "content_addressed_artifact_cache": MagicMock(),
        "verbose": False,
    }
    return arguments


def test_single_candidate_preserves_settings_and_callbacks() -> None:
    package = _package()
    arguments = _load_arguments()
    model = mock.sentinel.model
    with mock.patch.object(
        core, "initialize_model", return_value=(model, "/cache/model")
    ) as initialize:
        result = core.load_model_package_candidate(package, **arguments)

    assert result is model
    initialize.assert_called_once()
    forwarded = initialize.call_args.kwargs
    assert forwarded["model_package"] is package
    assert forwarded["model_init_kwargs"] == arguments["model_init_kwargs"]
    assert forwarded["model_init_kwargs"] is not arguments["model_init_kwargs"]
    assert forwarded["requested_model_id"] == "model-alias"
    assert forwarded["model_id"] == "canonical/1"
    assert forwarded["verify_hash_while_download"] is True
    assert forwarded["download_files_without_hash"] is False
    assert (
        forwarded["content_addressed_artifact_cache"]
        is arguments["content_addressed_artifact_cache"]
    )
    forwarded["on_file_created"]("/cache/model/weights")
    callback = arguments["model_access_manager"].on_file_created.call_args
    assert callback.kwargs["access_identifiers"].package_id == package.package_id
    arguments["model_access_manager"].on_model_loaded.assert_called_once()
    arguments["point_model_directory"].assert_called_once_with("/cache/model")


def test_single_candidate_propagates_original_error_without_fallback() -> None:
    error = RuntimeError("initialization failed")
    arguments = _load_arguments()
    with mock.patch.object(core, "initialize_model", side_effect=error) as initialize:
        with pytest.raises(RuntimeError) as captured:
            core.load_model_package_candidate(_package(), **arguments)

    assert captured.value is error
    initialize.assert_called_once()
    arguments["model_access_manager"].on_model_loaded.assert_not_called()
    arguments["point_model_directory"].assert_not_called()


def test_ordinary_loader_retains_ranked_fallback_and_fresh_kwargs() -> None:
    first, second = _package("first"), _package("second")
    arguments = _load_arguments()
    calls = []

    def initialize(**kwargs):
        calls.append(kwargs["model_package"])
        assert kwargs["model_init_kwargs"]["trt_cuda_graph_cache"] is True
        kwargs["model_init_kwargs"]["trt_cuda_graph_cache"] = False
        if kwargs["model_package"] is first:
            raise RuntimeError("first failed")

        return mock.sentinel.model, "/cache/second"

    with mock.patch.object(core, "initialize_model", side_effect=initialize):
        result = core.attempt_loading_matching_model_packages(
            matching_model_packages=[first, second], **arguments
        )

    assert result is mock.sentinel.model
    assert calls == [first, second]
    assert arguments["model_init_kwargs"]["trt_cuda_graph_cache"] is True


def test_ordinary_loader_retains_attempt_limit_and_error_list() -> None:
    error = RuntimeError("first failed")
    with mock.patch.object(core, "initialize_model", side_effect=error) as initialize:
        with pytest.raises(ModelPackageAlternativesExhaustedError) as captured:
            core.attempt_loading_matching_model_packages(
                matching_model_packages=[_package("first"), _package("second")],
                max_package_loading_attempts=1,
                **_load_arguments(),
            )

    initialize.assert_called_once()
    assert captured.value.alternatives_errors == [error]


@pytest.mark.parametrize("strict", [False, True])
def test_explicit_package_runtime_recheck_is_opt_in(strict: bool) -> None:
    selected, alternative = _package(), _package("alternative")
    rejected = auto_negotiation.DiscardedPackage(
        package_id=selected.package_id, reason="Different worker runtime"
    )
    with mock.patch.object(
        auto_negotiation,
        "remove_packages_not_matching_implementation",
        side_effect=lambda **kwargs: (kwargs["model_packages"], []),
    ), mock.patch.object(
        auto_negotiation,
        "filter_model_packages_matching_runtime_environment",
        return_value=([], [rejected]),
    ) as runtime_filter:
        arguments = {
            "model_architecture": "rfdetr",
            "task_type": "instance-segmentation",
            "model_packages": [selected, alternative],
            "requested_model_package_id": selected.package_id,
            "device": torch.device("cpu"),
            "require_compatible_explicit_package": strict,
        }
        if strict:
            with pytest.raises(NoModelPackagesAvailableError):
                auto_negotiation.negotiate_model_packages(**arguments)

            assert runtime_filter.call_args.kwargs["model_packages"] == [selected]
        else:
            result = auto_negotiation.negotiate_model_packages(**arguments)
            assert result == [selected]
            runtime_filter.assert_not_called()


def test_explicit_package_recheck_cannot_choose_other_compatible_package() -> None:
    selected, alternative = _package(), _package("alternative")
    with mock.patch.object(
        auto_negotiation,
        "remove_packages_not_matching_implementation",
        side_effect=lambda **kwargs: (kwargs["model_packages"], []),
    ), mock.patch.object(
        auto_negotiation,
        "filter_model_packages_matching_runtime_environment",
        side_effect=lambda **kwargs: (kwargs["model_packages"], []),
    ) as runtime_filter:
        result = auto_negotiation.negotiate_model_packages(
            model_architecture="rfdetr",
            task_type="instance-segmentation",
            model_packages=[alternative, selected],
            requested_model_package_id=selected.package_id,
            require_compatible_explicit_package=True,
            device=torch.device("cpu"),
        )

    assert result == [selected]
    assert runtime_filter.call_args.kwargs["model_packages"] == [selected]
