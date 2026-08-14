from unittest.mock import MagicMock

import torch

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.models.auto_loaders.entities import BackendType
from inference_models.models.auto_loaders.models_registry import (
    REGISTERED_MODELS,
    VLM_TASK,
)
from inference_models.models.qwen3_5.qwen3_5_hf import Qwen35HF
from inference_models.models.qwen3_8.qwen3_8_hf import Qwen38HF


def test_qwen3_8_registry_entry_resolves_to_qwen38hf() -> None:
    # when
    registered_class = REGISTERED_MODELS[
        ("qwen3_8", VLM_TASK, BackendType.HF)
    ].resolve()

    # then
    assert registered_class is Qwen38HF


def test_qwen38hf_inherits_qwen35hf_processing() -> None:
    # Qwen3.8 ships the qwen3_5 architecture - the class must reuse the
    # Qwen35HF loading / pre- / post-processing implementation.
    assert issubclass(Qwen38HF, Qwen35HF)
    assert Qwen38HF.pre_process_generation is Qwen35HF.pre_process_generation
    assert Qwen38HF.post_process_generation is Qwen35HF.post_process_generation
    assert Qwen38HF.from_pretrained.__func__ is Qwen35HF.from_pretrained.__func__


def test_generate_applies_qwen3_8_default_max_new_tokens() -> None:
    # given
    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
    processor = MagicMock()
    qwen = Qwen38HF(
        model=model,
        processor=processor,
        inference_config=None,
        device=torch.device("cpu"),
    )
    inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

    # when
    result = qwen.generate(inputs=inputs)

    # then
    assert (
        model.generate.call_args.kwargs["max_new_tokens"]
        == INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
    )
    assert result.shape == (1, 3)


def test_generate_respects_explicit_max_new_tokens() -> None:
    # given
    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
    processor = MagicMock()
    qwen = Qwen38HF(
        model=model,
        processor=processor,
        inference_config=None,
        device=torch.device("cpu"),
    )
    inputs = {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

    # when
    _ = qwen.generate(inputs=inputs, max_new_tokens=17)

    # then
    assert model.generate.call_args.kwargs["max_new_tokens"] == 17
