from unittest.mock import MagicMock

import numpy as np
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.glm_ocr.glm_ocr_hf import GlmOcrHF


def test_generate_uses_default_max_new_tokens_when_none_is_given() -> None:
    model = MagicMock()
    model.generate.return_value = np.array([[11, 12, 21, 22]])
    glm_ocr = GlmOcrHF(model=model, processor=MagicMock(), device=MagicMock())

    result = glm_ocr.generate(
        inputs={"input_ids": np.array([[11, 12]])},
        max_new_tokens=None,
    )

    assert model.generate.call_args.kwargs["max_new_tokens"] == (
        INFERENCE_MODELS_GLM_OCR_DEFAULT_MAX_NEW_TOKENS
    )
    assert result.tolist() == [[21, 22]]


def test_pre_process_generation_drops_token_type_ids() -> None:
    # given - the GLM checkpoint ships a bare `PreTrainedTokenizerFast`, whose
    # default `model_input_names` makes `apply_chat_template` emit
    # `token_type_ids` - the model does not consume it and the dynamic batcher
    # cannot collate it
    input_ids = torch.tensor([[11, 12]], dtype=torch.long)
    processor = MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "token_type_ids": torch.zeros_like(input_ids),
        "pixel_values": torch.randn(4, 8),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }
    glm_ocr = GlmOcrHF(
        model=MagicMock(), processor=processor, device=torch.device("cpu")
    )

    # when
    result = glm_ocr.pre_process_generation(
        images=np.zeros((2, 2, 3), dtype=np.uint8)
    )

    # then
    assert set(result.keys()) == {
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
    }
