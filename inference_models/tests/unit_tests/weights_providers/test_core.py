from unittest import mock

import pytest

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers import core
from inference_models.weights_providers.core import get_model_from_provider


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
@mock.patch.object(core, "OFFLINE_MODE", False)
def test_get_model_from_provider_when_provider_recognised() -> None:
    """A registered provider returns metadata when online mode is enabled."""
    # when
    result = get_model_from_provider(model_id="my-model", provider="some")

    # then
    assert result == "ok"


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
@mock.patch.object(core, "OFFLINE_MODE", False)
def test_get_model_from_provider_when_provider_not_recognised() -> None:
    """An unknown provider raises a model retrieval error."""
    # when
    with pytest.raises(ModelRetrievalError):
        _ = get_model_from_provider(model_id="my-model", provider="unknown")


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {})
@mock.patch.object(core, "OFFLINE_MODE", True)
def test_get_model_from_custom_provider_in_offline_mode() -> None:
    """A custom provider can resolve fully local metadata in offline mode."""
    # given
    local_metadata = object()
    local_provider = mock.Mock(return_value=local_metadata)
    core.register_model_provider("local", local_provider)

    # when
    result = get_model_from_provider(model_id="my-model", provider="local")

    # then
    assert result is local_metadata
    local_provider.assert_called_once_with("my-model", None)


@mock.patch.object(core, "OFFLINE_MODE", True)
def test_get_model_from_builtin_network_provider_in_offline_mode() -> None:
    """The built-in network provider remains unavailable in offline mode."""
    # when
    with pytest.raises(ModelRetrievalError, match="OFFLINE_MODE"):
        get_model_from_provider(model_id="my-model", provider="roboflow")


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {})
@mock.patch.object(core, "OFFLINE_MODE", True)
def test_custom_provider_cannot_override_builtin_name_offline() -> None:
    """Built-in provenance cannot be replaced by a self-asserted local handler."""
    # given
    local_provider = mock.Mock()
    with pytest.raises(ValueError, match="reserved"):
        core.register_model_provider("roboflow", local_provider)

    # then
    assert core.WEIGHTS_PROVIDERS == {}
    local_provider.assert_not_called()


@mock.patch.object(
    core,
    "WEIGHTS_PROVIDERS",
    {"roboflow": lambda model_id, api_key: object()},
)
def test_builtin_network_requirement_is_bound_to_reserved_provider_name() -> None:
    assert core.model_provider_requires_network(provider="roboflow") is True
