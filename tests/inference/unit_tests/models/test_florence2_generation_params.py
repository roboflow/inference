from types import SimpleNamespace

import pytest

from inference.models.florence2.florence2 import Florence2, LoRAFlorence2


@pytest.mark.parametrize("model_class", [Florence2, LoRAFlorence2])
def test_florence2_generation_forbids_a_repeated_bos(model_class) -> None:
    # given
    model = model_class.__new__(model_class)
    model.processor = SimpleNamespace(tokenizer=SimpleNamespace(bos_token_id=0))
    inputs = {"input_ids": "ids", "pixel_values": "pixels", "attention_mask": "mask"}

    # when
    params = model.prepare_generation_params(preprocessed_inputs=inputs)

    # then
    assert params == {
        "input_ids": "ids",
        "pixel_values": "pixels",
        "bad_words_ids": [[0, 0]],
    }
