from types import SimpleNamespace

import pytest
import torch

from inference_models.errors import UnauthorizedModelAccessError
from inference_models.weights_providers.entities import ModelDependency

from inference_model_manager import shared_base_resolution as shared_base
from inference_model_manager.shared_base_resolution import (
    SharedBaseResolution,
    derive_base_key,
    resolve_shared_base,
)


def _head_metadata(dependencies):
    return SimpleNamespace(
        model_id="head/1",
        model_architecture="roboflow-instant",
        task_type="object-detection",
        model_packages=[],
        model_dependencies=dependencies,
    )


def _base_metadata():
    return SimpleNamespace(
        model_id="owlv2",
        model_architecture="owlv2",
        task_type="object-detection",
        model_packages=[SimpleNamespace(package_id="owlv2-pkg-a")],
        model_dependencies=None,
    )


def _owlv2_dep(package_id=None):
    return ModelDependency(
        name="feature_extractor", model_id="owlv2", model_package_id=package_id
    )


def _patch_provider(monkeypatch, metadata_by_id, calls=None):
    def fake_fetch(provider, model_id, api_key=None, **kwargs):
        if calls is not None:
            calls.append(model_id)
        return metadata_by_id[model_id]

    monkeypatch.setattr(shared_base, "get_model_from_provider", fake_fetch)


def _patch_negotiate(monkeypatch, packages, recorder=None):
    def fake_negotiate(**kwargs):
        if recorder is not None:
            recorder.update(kwargs)
        return packages

    monkeypatch.setattr(shared_base, "negotiate_model_packages", fake_negotiate)


def test_resolve_returns_none_when_no_dependencies(monkeypatch):
    _patch_provider(monkeypatch, {"head/1": _head_metadata(None)})
    _patch_negotiate(monkeypatch, [])

    assert resolve_shared_base("head/1", "key", "cpu", {}) is None


def test_resolve_returns_none_when_no_supported_dependency(monkeypatch):
    other = ModelDependency(name="x", model_id="resnet", model_package_id="p")
    _patch_provider(monkeypatch, {"head/1": _head_metadata([other])})
    _patch_negotiate(monkeypatch, [])

    assert resolve_shared_base("head/1", "key", "cpu", {}) is None


def test_resolves_shared_base(monkeypatch):
    recorder = {}
    _patch_provider(
        monkeypatch,
        {"head/1": _head_metadata([_owlv2_dep()]), "owlv2": _base_metadata()},
    )
    _patch_negotiate(
        monkeypatch, [SimpleNamespace(package_id="owlv2-pkg-a")], recorder
    )

    detection = resolve_shared_base("head/1", "key", "cpu", {})

    assert isinstance(detection, SharedBaseResolution)
    assert detection.dep_name == "feature_extractor"
    assert detection.dep_model_id == "owlv2"
    assert detection.dep_metadata_package_id is None
    assert detection.resolved_package_id == "owlv2-pkg-a"
    assert detection.base_key
    # negotiation runs with the dependency's (unset) requested package id.
    assert recorder["requested_model_package_id"] is None
    assert recorder["device"] == torch.device("cpu")


def test_resolve_returns_none_when_negotiation_yields_no_package(monkeypatch):
    _patch_provider(
        monkeypatch,
        {"head/1": _head_metadata([_owlv2_dep()]), "owlv2": _base_metadata()},
    )
    _patch_negotiate(monkeypatch, [])

    assert resolve_shared_base("head/1", "key", "cpu", {}) is None


def test_resolve_propagates_auth_error(monkeypatch):
    def fake_fetch(provider, model_id, api_key=None, **kwargs):
        raise UnauthorizedModelAccessError(message="denied")

    monkeypatch.setattr(shared_base, "get_model_from_provider", fake_fetch)

    with pytest.raises(UnauthorizedModelAccessError):
        resolve_shared_base("head/1", "key", "cpu", {})


def test_resolve_fails_open_on_generic_error(monkeypatch):
    def fake_fetch(provider, model_id, api_key=None, **kwargs):
        raise RuntimeError("provider offline")

    monkeypatch.setattr(shared_base, "get_model_from_provider", fake_fetch)

    assert resolve_shared_base("head/1", "key", "cpu", {}) is None


def test_resolve_caches_metadata(monkeypatch):
    calls = []
    _patch_provider(
        monkeypatch,
        {"head/1": _head_metadata([_owlv2_dep()]), "owlv2": _base_metadata()},
        calls,
    )
    _patch_negotiate(monkeypatch, [SimpleNamespace(package_id="owlv2-pkg-a")])
    cache = {}

    resolve_shared_base("head/1", "key", "cpu", cache)
    resolve_shared_base("head/1", "key", "cpu", cache)

    # head + base fetched once each, second call served from cache.
    assert calls == ["head/1", "owlv2"]


def test_base_key_excludes_api_key_for_whitelisted_base():
    device = torch.device("cpu")
    key_a = derive_base_key("owlv2", "pkg-a", device, api_key="tenant-a")
    key_b = derive_base_key("owlv2", "pkg-a", device, api_key="tenant-b")

    assert key_a == key_b


def test_base_key_includes_api_key_for_non_whitelisted_base():
    device = torch.device("cpu")
    key_a = derive_base_key("secret-base", "pkg-a", device, api_key="tenant-a")
    key_b = derive_base_key("secret-base", "pkg-a", device, api_key="tenant-b")

    assert key_a != key_b


def test_base_key_changes_with_resolved_package():
    device = torch.device("cpu")
    key_a = derive_base_key("owlv2", "pkg-a", device, api_key="k")
    key_b = derive_base_key("owlv2", "pkg-b", device, api_key="k")

    assert key_a != key_b
