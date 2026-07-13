from unittest.mock import MagicMock

import pytest

from inference.core.exceptions import ModelNotRecognisedError
from inference.core.registries.base import ModelRegistry
from inference.core.roboflow_api import ModelEndpointType


def test_getting_model_on_registry_hit() -> None:
    # given
    model_mock = MagicMock()
    registry = ModelRegistry(registry_dict={"yolov8n": model_mock})

    # when
    result = registry.get_model(model_type="yolov8n", model_id="non-important")

    # then
    assert result is model_mock


def test_getting_model_on_registry_miss() -> None:
    registry = ModelRegistry(registry_dict={})

    # when
    with pytest.raises(ModelNotRecognisedError):
        _ = registry.get_model(model_type="yolov8n", model_id="non-important")


def test_default_model_auth_target_is_requested_model() -> None:
    registry = ModelRegistry(registry_dict={})

    result = registry.get_model_auth_targets(
        model_id="some-model/1",
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )

    assert result == ["some-model/1"]
