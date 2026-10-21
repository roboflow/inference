"""Admission preparation must never allocate a model or silently change package."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from inference_models.models.auto_loaders import core
from inference_models.models.auto_loaders.admission import (
    AdmissionCandidate,
    load_admitted_candidate,
    prepare_model_packages,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.weights_providers.entities import (
    ModelMetadata,
    ModelPackageMetadata,
)


@pytest.fixture
def path(monkeypatch):
    packages = [
        ModelPackageMetadata(
            package_id=name,
            backend=BackendType.ONNX,
            package_artefacts=[],
            trusted_source=True,
        )
        for name in ["first", "second"]
    ]
    metadata = ModelMetadata(
        model_id="canonical/1",
        model_architecture="yolov8",
        model_packages=packages,
        task_type="object-detection",
    )
    provider = MagicMock(return_value=metadata)
    negotiation = MagicMock(return_value=packages)
    initializer = MagicMock()
    cache = MagicMock()
    cache.retrieve.return_value = None
    for name, value in [
        ("get_model_from_provider", provider),
        ("negotiate_model_packages", negotiation),
        ("load_model_package_candidate", initializer),
        ("x_ray_runtime_environment", MagicMock()),
        ("_runtime_compatibility_content", MagicMock(return_value={})),
        ("OFFLINE_MODE", False),
        ("OFFLINE_MODE_WARM_UP", False),
    ]:
        monkeypatch.setattr(core, name, value)
    monkeypatch.setattr(
        core,
        "attempt_loading_model_with_auto_load_cache",
        MagicMock(side_effect=AssertionError("cache must not load")),
    )
    monkeypatch.setattr(
        core,
        "attempt_loading_matching_model_packages",
        MagicMock(side_effect=AssertionError("fallback must not run in worker")),
    )
    return SimpleNamespace(
        packages=packages,
        metadata=metadata,
        provider=provider,
        negotiation=negotiation,
        initializer=initializer,
        cache=cache,
    )


def test_preparation_only_returns_metadata_and_preserves_cached_rank(path, monkeypatch):
    path.cache.retrieve.return_value = SimpleNamespace(
        model_id="alias", canonical_model_id="canonical/1", model_package_id="second"
    )
    monkeypatch.setattr(
        core, "_verified_auto_cache_package_dir", MagicMock(return_value="/verified")
    )
    prepared = prepare_model_packages(
        "alias", api_key="secret", auto_resolution_cache=path.cache
    )
    assert [p.package_id for p in prepared.candidates] == ["second", "first"]
    path.initializer.assert_not_called()
    assert (
        path.negotiation.call_args.kwargs["require_compatible_explicit_package"] is True
    )


def test_worker_reauthorizes_and_loads_only_admitted_candidate(path):
    path.negotiation.return_value = [path.packages[1]]
    result = load_admitted_candidate(
        "alias",
        candidate=AdmissionCandidate("canonical/1", "second"),
        api_key="secret",
        auto_resolution_cache=path.cache,
    )
    assert result is path.initializer.return_value
    assert path.initializer.call_args.args[0].package_id == "second"
    assert path.negotiation.call_args.kwargs["requested_model_package_id"] == "second"
    assert (
        path.negotiation.call_args.kwargs["require_compatible_explicit_package"] is True
    )
    path.provider.assert_called_once()


def test_worker_propagates_initialization_failure_for_manager_fallback(path):
    failure = RuntimeError("initializer failed")
    path.negotiation.return_value = [path.packages[0]]
    path.initializer.side_effect = failure
    with pytest.raises(RuntimeError) as error:
        load_admitted_candidate(
            "alias",
            candidate=AdmissionCandidate("canonical/1", "first"),
            api_key="secret",
            auto_resolution_cache=path.cache,
        )
    assert error.value is failure
    assert error.value._admission_candidate_failed
    path.initializer.assert_called_once()


def test_changed_alias_or_candidate_fails_before_initialization(path):
    for candidate in [
        AdmissionCandidate("different/2", "first"),
        AdmissionCandidate("canonical/1", "first"),
    ]:
        with pytest.raises(ValueError):
            load_admitted_candidate(
                "alias",
                candidate=candidate,
                api_key="secret",
                auto_resolution_cache=path.cache,
            )
    path.initializer.assert_not_called()


def test_offline_and_dependency_paths_are_explicitly_unsupported(path, monkeypatch):
    monkeypatch.setattr(core, "OFFLINE_MODE", True)
    with pytest.raises(NotImplementedError):
        prepare_model_packages("alias", api_key="secret")
    path.provider.assert_not_called()


def test_explicit_package_cannot_be_replaced(path):
    with pytest.raises(ValueError, match="Explicit requested package"):
        load_admitted_candidate(
            "alias",
            candidate=AdmissionCandidate("canonical/1", "first"),
            api_key="secret",
            model_package_id="second",
        )
    path.provider.assert_not_called()


def test_admitted_load_preserves_original_resolution_cache_key(path):
    prepare_model_packages("alias", api_key="secret", auto_resolution_cache=path.cache)
    original_key = path.cache.retrieve.call_args.kwargs["auto_negotiation_hash"]
    path.negotiation.return_value = [path.packages[1]]
    load_admitted_candidate(
        "alias",
        candidate=AdmissionCandidate("canonical/1", "second"),
        api_key="secret",
        auto_resolution_cache=path.cache,
    )
    assert path.initializer.call_args.kwargs["auto_negotiation_hash"] == original_key
