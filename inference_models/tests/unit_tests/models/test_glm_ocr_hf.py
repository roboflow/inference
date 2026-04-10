from unittest.mock import MagicMock

import numpy as np

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
