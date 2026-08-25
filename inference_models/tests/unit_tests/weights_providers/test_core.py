from unittest import mock

import pytest

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers import core
from inference_models.weights_providers.core import get_model_from_provider


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
def test_get_model_from_provider_when_provider_recognised() -> None:
    """A registered provider returns metadata."""
    # when
    result = get_model_from_provider(model_id="my-model", provider="some")

    # then
    assert result == "ok"


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
def test_get_model_from_provider_when_provider_not_recognised() -> None:
    """An unknown provider raises a model retrieval error."""
    # when
    with pytest.raises(ModelRetrievalError):
        _ = get_model_from_provider(model_id="my-model", provider="unknown")


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {})
def test_registered_custom_provider_serves_metadata() -> None:
    """A custom provider resolves metadata through the registry."""
    # given
    local_metadata = object()
    local_provider = mock.Mock(return_value=local_metadata)
    core.register_model_provider("local", local_provider)

    # when
    result = get_model_from_provider(model_id="my-model", provider="local")

    # then
    assert result is local_metadata
    local_provider.assert_called_once_with("my-model", None)


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {})
@pytest.mark.parametrize(
    "reserved_name", ["roboflow", "Roboflow", "roboflow-offline-weights"]
)
def test_custom_provider_cannot_override_reserved_names(reserved_name: str) -> None:
    """Built-in provenance cannot be replaced by a self-asserted local handler."""
    # given
    local_provider = mock.Mock()
    with pytest.raises(ValueError, match="reserved"):
        core.register_model_provider(reserved_name, local_provider)

    # then
    assert core.WEIGHTS_PROVIDERS == {}
    local_provider.assert_not_called()
