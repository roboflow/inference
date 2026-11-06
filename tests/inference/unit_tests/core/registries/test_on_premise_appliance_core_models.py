from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import RoboflowAPINotAuthorizedError
from inference.core.managers import base as base_module
from inference.core.managers.base import ModelManager
from inference.core.registries import roboflow as registry_module
from inference.core.registries.roboflow import (
    _check_if_api_key_has_access_to_model,
    is_generic_core_model_id,
)

APPLIANCE_PINNED_MODELS = [
    "sam3/sam3_final",
    "sam2/hiera_tiny",
    "owlv2/owlv2-large-patch14-ensemble",
    "clip/ViT-L-14",
]


@pytest.fixture(autouse=True)
def _fresh_access_cache():
    _check_if_api_key_has_access_to_model.cache_clear()
    yield
    _check_if_api_key_has_access_to_model.cache_clear()


@pytest.fixture
def platform_calls(monkeypatch):
    """Record every model access call to the platform; refuse other keys."""
    calls = []

    def model_data(api_key, model_id, **kwargs):
        calls.append((api_key, model_id))
        if api_key != "owner-key":
            raise RoboflowAPINotAuthorizedError("denied")
        return {"ort": {}}

    monkeypatch.setattr(registry_module, "USE_INFERENCE_MODELS", False)
    monkeypatch.setattr(registry_module, "get_roboflow_model_data", model_data)
    monkeypatch.setattr(registry_module, "get_roboflow_instant_model_data", model_data)
    monkeypatch.setattr(
        registry_module, "get_model_metadata_from_inference_models_registry", model_data
    )
    monkeypatch.setattr(base_module, "MODELS_CACHE_AUTH_ENABLED", True)
    monkeypatch.setattr(base_module, "OFFLINE_MODE", False)

    return calls


@pytest.mark.parametrize(
    "model_id",
    APPLIANCE_PINNED_MODELS
    + ["clip", "sam3/sam3_interactive", "depth-anything-v2/small", "yolo_world/v2-s"],
)
def test_core_model_ids_are_generic(model_id) -> None:
    assert is_generic_core_model_id(model_id=model_id) is True


@pytest.mark.parametrize(
    "model_id",
    [
        "project-a/1",
        "workspace/project/1",
        "yolov8n-640",
        "pp-ocrv6-det/small",
        "paligemma/not-a-version",
        "",
        None,
    ],
)
def test_workspace_and_alias_model_ids_are_not_generic(model_id) -> None:
    assert is_generic_core_model_id(model_id=model_id) is False


@pytest.mark.parametrize("model_id", APPLIANCE_PINNED_MODELS)
def test_appliance_preload_of_core_models_needs_no_key(
    monkeypatch, platform_calls, model_id
) -> None:
    # DF-118: with MODELS_CACHE_AUTH_ENABLED the keyless preload of the pinned
    # core models failed, so /model/registry stayed empty.
    monkeypatch.setattr(registry_module, "ON_PREMISE_APPLIANCE", True)
    model_manager = ModelManager(model_registry=MagicMock())

    model_manager.add_model(model_id=model_id, api_key=None)

    assert model_id in model_manager.models()
    assert platform_calls == []


def test_without_appliance_mode_keyless_core_preload_is_refused(
    monkeypatch, platform_calls
) -> None:
    monkeypatch.setattr(registry_module, "ON_PREMISE_APPLIANCE", False)
    model_manager = ModelManager(model_registry=MagicMock())

    with pytest.raises(RoboflowAPINotAuthorizedError):
        model_manager.add_model(model_id="sam3/sam3_final", api_key=None)

    assert platform_calls == [(None, "sam3/sam3_final")]


def test_appliance_mode_checks_every_request_for_a_loaded_workspace_model(
    monkeypatch, platform_calls
) -> None:
    monkeypatch.setattr(registry_module, "ON_PREMISE_APPLIANCE", True)
    model_manager = ModelManager(model_registry=MagicMock())
    model_manager.add_model(model_id="project-a/1", api_key="owner-key")

    with pytest.raises(RoboflowAPINotAuthorizedError):
        model_manager.add_model(model_id="project-a/1", api_key="other-workspace-key")

    assert platform_calls == [
        ("owner-key", "project-a/1"),
        ("other-workspace-key", "project-a/1"),
    ]


def test_appliance_mode_core_model_with_any_key_makes_no_platform_call(
    monkeypatch, platform_calls
) -> None:
    monkeypatch.setattr(registry_module, "ON_PREMISE_APPLIANCE", True)

    allowed = _check_if_api_key_has_access_to_model(
        api_key="other-workspace-key",
        model_id="clip/ViT-B-16",
        endpoint_type="core_model",
    )

    assert allowed is True
    assert platform_calls == []
