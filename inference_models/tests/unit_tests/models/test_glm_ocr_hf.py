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


def test_pre_process_generation_casts_floating_point_inputs_to_model_dtype() -> None:
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    processor = MagicMock()
    processor.apply_chat_template.return_value = {
        "input_ids": torch.tensor([[1, 2]], dtype=torch.int64),
        "pixel_values": torch.tensor([[[1.0]]], dtype=torch.float32),
    }
    glm_ocr = GlmOcrHF(
        model=model,
        processor=processor,
        device=torch.device("cpu"),
    )

    inputs = glm_ocr.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert inputs["input_ids"].dtype == torch.int64
    assert inputs["pixel_values"].dtype == torch.bfloat16
