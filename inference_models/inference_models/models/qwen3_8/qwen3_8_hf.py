from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers
from packaging.version import Version

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat
from inference_models.errors import EnvironmentConfigurationError
from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF

# Qwen3.8 reuses the qwen3_5 architecture class, so an older transformers
# imports cleanly but fails deep inside processor/chat-template handling -
# guard explicitly instead of relying on an ImportError that never comes.
MINIMUM_TRANSFORMERS_VERSION = "5.8.0"


def _ensure_supported_transformers_version() -> None:
    installed = Version(transformers.__version__)
    if installed < Version(MINIMUM_TRANSFORMERS_VERSION):
        raise EnvironmentConfigurationError(
            f"Qwen3.8 requires transformers>={MINIMUM_TRANSFORMERS_VERSION} "
            f"(installed: {transformers.__version__}). Upgrade the "
            "`transformers` package to load this model."
        )


class Qwen38HF(Qwen35HF):
    # Qwen3.8 ships the qwen3_5 architecture (config.json declares
    # Qwen3_5ForConditionalGeneration / model_type "qwen3_5"), so loading and
    # pre/post-processing are inherited from Qwen35HF; only the generation
    # defaults and the transformers floor are family-specific.

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "Qwen38HF":
        _ensure_supported_transformers_version()
        return super().from_pretrained(model_name_or_path, **kwargs)

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        # Qwen35HF.prompt() forwards its own hardcoded defaults to generate(),
        # which would bypass the qwen3_8 env-derived defaults - resolve them
        # here so every entry point behaves the same.
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
        if do_sample is None:
            do_sample = INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE
        return super().prompt(
            images=images,
            prompt=prompt,
            input_color_format=input_color_format,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            skip_special_tokens=skip_special_tokens,
            enable_thinking=enable_thinking,
            **kwargs,
        )

    def generate(
        self,
        inputs: dict,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
        if do_sample is None:
            do_sample = INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE
        return super().generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
