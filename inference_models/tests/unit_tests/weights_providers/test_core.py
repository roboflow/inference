from unittest import mock

import pytest

from inference_models.errors import ModelRetrievalError
from inference_models.weights_providers import core
from inference_models.weights_providers.core import get_model_from_provider


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
def test_get_model_from_provider_when_provider_recognised() -> None:
    # when
    result = get_model_from_provider(model_id="my-model", provider="some")

    # then
    assert result == "ok"


@mock.patch.object(core, "WEIGHTS_PROVIDERS", {"some": lambda model_id, api_key: "ok"})
def test_get_model_from_provider_when_provider_not_recognised() -> None:
    # when
    with pytest.raises(ModelRetrievalError):
        _ = get_model_from_provider(model_id="my-model", provider="unknown")
