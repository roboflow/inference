from types import SimpleNamespace

import pytest

from inference.models.florence2.florence2 import Florence2, LoRAFlorence2


def _model(model_class, *, bos_token_id, decoder_start_token_id=2):
    model = model_class.__new__(model_class)
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=bos_token_id))
    model.model = SimpleNamespace(
        generation_config=SimpleNamespace(decoder_start_token_id=decoder_start_token_id)
    )
    return model


INPUTS = {"input_ids": "ids", "pixel_values": "pixels", "attention_mask": "mask"}


@pytest.mark.parametrize("model_class", [Florence2, LoRAFlorence2])
def test_florence2_generation_forbids_a_repeated_bos(model_class) -> None:
    # given
    model = _model(model_class, bos_token_id=0)

    # when
    params = model.prepare_generation_params(preprocessed_inputs=INPUTS)

    # then
    assert params == {
        "input_ids": "ids",
        "pixel_values": "pixels",
        "bad_words_ids": [[0, 0]],
    }


@pytest.mark.parametrize("model_class", [Florence2, LoRAFlorence2])
def test_florence2_generation_skips_the_ban_without_a_bos_token(model_class) -> None:
    # given
    model = _model(model_class, bos_token_id=None)

    # when
    params = model.prepare_generation_params(preprocessed_inputs=INPUTS)

    # then
    assert params == {"input_ids": "ids", "pixel_values": "pixels"}


@pytest.mark.parametrize("model_class", [Florence2, LoRAFlorence2])
def test_florence2_generation_skips_the_ban_when_decoding_starts_on_bos(model_class) -> None:
    # given
    model = _model(model_class, bos_token_id=0, decoder_start_token_id=0)

    # when
    params = model.prepare_generation_params(preprocessed_inputs=INPUTS)

    # then
    assert params == {"input_ids": "ids", "pixel_values": "pixels"}
