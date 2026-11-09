from unittest.mock import MagicMock

import pytest
import torch

from inference.models.transformers import transformers


def _model() -> transformers.TransformerModel:
    model = transformers.TransformerModel.__new__(transformers.TransformerModel)
    model.generation_includes_input = True
    model.skip_special_tokens = True
    model_inputs = MagicMock()
    model_inputs.__getitem__.return_value = torch.zeros((1, 3))
    model_inputs.keys.return_value = []
    model.processor = MagicMock()
    model.processor.return_value.to.return_value = model_inputs
    model.processor.decode.return_value = "text"
    model.model = MagicMock()
    model.model.generate.return_value = torch.zeros((1, 5), dtype=torch.long)
    return model


def test_predict_uses_the_request_max_new_tokens() -> None:
    # given
    model = _model()

    # when
    model.predict("image", prompt="caption", max_new_tokens=17)

    # then
    assert model.model.generate.call_args.kwargs["max_new_tokens"] == 17


@pytest.mark.parametrize("request_value", [None, 0])
def test_predict_falls_back_to_the_configured_default(
    monkeypatch, request_value
) -> None:
    # given
    monkeypatch.setattr(transformers, "LMM_DEFAULT_MAX_NEW_TOKENS", 256)
    model = _model()

    # when
    model.predict("image", prompt="caption", max_new_tokens=request_value)

    # then
    assert model.model.generate.call_args.kwargs["max_new_tokens"] == 256
