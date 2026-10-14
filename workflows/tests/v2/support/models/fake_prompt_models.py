"""Deterministic stand-ins for the ``inference_models`` local VLM ``prompt`` API.

Each ``Fake*`` copies the parameter names and defaults of one backend's
``prompt`` (``Qwen25VLHF``, ``Qwen3VLHF``, ``Qwen35HF``, ``SmolVLMHF``). It is
NOT model inference and does not parse prompts: it records every call and
answers each image with text that echoes what reached it::

    answer = "<Fake class>|<prompt received>|<H>x<W>|<mean pixel>|<max_new_tokens>"

``tests/v2/unit/blocks/models/test_vlm.py`` checks the copied signatures
against the backend sources, so a backend change fails a test.
"""

from typing import Any, Dict, List, Optional

import torch
from roboflow_workflows.v2.blocks.models.cache import ModelRequest

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS,
    INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS,
)


class FakePromptModel:
    """Records ``prompt`` calls and answers each image with an echo."""

    def __init__(self, request: Optional[ModelRequest] = None):
        self.request = request
        self.calls: List[Dict[str, Any]] = []

    def _answer(
        self, images: List[torch.Tensor], *, prompt: Any, max_new_tokens: Any, **kw
    ) -> List[str]:
        self.calls.append(
            {"images": images, "prompt": prompt, "max_new_tokens": max_new_tokens, **kw}
        )
        answers = []
        for image in images:
            _, height, width = image.shape
            mean = round(image.float().mean().item())
            answers.append(
                f"{type(self).__name__}|{prompt}|{height}x{width}|{mean}|"
                f"{max_new_tokens}"
            )

        return answers


class FakeQwen25VL(FakePromptModel):
    """``Qwen25VLHF.prompt`` parameters."""

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: int = INFERENCE_MODELS_QWEN25_VL_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_QWEN25_VL_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = (
            INFERENCE_MODELS_QWEN25_VL_DEFAULT_SKIP_SPECIAL_TOKENS
        ),
        pre_processing_overrides: Any = None,
        **kwargs,
    ) -> List[str]:
        answers = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            do_sample=do_sample,
            kwargs=kwargs,
        )

        return answers


class FakeQwen3VL(FakePromptModel):
    """``Qwen3VLHF.prompt`` parameters."""

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        answers = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            do_sample=do_sample,
            kwargs=kwargs,
        )

        return answers


class FakeQwen35(FakePromptModel):
    """``Qwen35HF.prompt`` parameters; thinking mode answers with dicts."""

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: str = None,
        input_color_format: str = None,
        max_new_tokens: int = 512,
        do_sample: bool = False,
        skip_special_tokens: bool = True,
        enable_thinking: bool = False,
        **kwargs,
    ) -> List[Any]:
        answers = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            do_sample=do_sample,
            enable_thinking=enable_thinking,
            kwargs=kwargs,
        )
        if enable_thinking:
            answers = [{"thinking": "", "answer": answer} for answer in answers]

        return answers


class FakeSmolVLM2(FakePromptModel):
    """``SmolVLMHF.prompt`` parameters."""

    def prompt(
        self,
        images: List[torch.Tensor],
        prompt: Optional[str] = None,
        images_to_single_prompt: bool = True,
        input_color_format: Optional[str] = None,
        max_new_tokens: int = INFERENCE_MODELS_SMOL_VLM_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_SMOL_VLM_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = (
            INFERENCE_MODELS_SMOL_VLM_DEFAULT_SKIP_SPECIAL_TOKENS
        ),
        pre_processing_overrides: Any = None,
        **kwargs,
    ) -> List[str]:
        answers = self._answer(
            images,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            input_color_format=input_color_format,
            do_sample=do_sample,
            images_to_single_prompt=images_to_single_prompt,
            kwargs=kwargs,
        )

        return answers


class FakePromptLoader:
    """``ModelCache`` loader building one ``model_type`` per request."""

    def __init__(self, model_type: type):
        self.model_type = model_type
        self.loaded: List[ModelRequest] = []
        self.models: List[FakePromptModel] = []

    def __call__(self, request: ModelRequest) -> FakePromptModel:
        self.loaded.append(request)
        model = self.model_type(request)
        self.models.append(model)

        return model
