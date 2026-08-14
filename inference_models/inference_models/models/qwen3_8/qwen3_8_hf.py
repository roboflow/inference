from typing import Optional

import torch

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF


class Qwen38HF(Qwen35HF):
    # Qwen3.8 ships the qwen3_5 architecture (config.json declares
    # Qwen3_5ForConditionalGeneration / model_type "qwen3_5"), so loading and
    # pre/post-processing are inherited from Qwen35HF. Requires
    # transformers>=5.8.0 at runtime for the Qwen3.8 tokenizer/chat template.

    def generate(
        self,
        inputs: dict,
        max_new_tokens: Optional[int] = None,
        do_sample: bool = INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
        return super().generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
