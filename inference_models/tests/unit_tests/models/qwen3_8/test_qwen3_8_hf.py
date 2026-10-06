from unittest.mock import MagicMock

import pytest
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE,
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


def test_generate_applies_qwen3_8_default_do_sample() -> None:
    # given
    model = MagicMock()
    model.generate.return_value = torch.zeros((1, 8), dtype=torch.long)
    qwen = Qwen38HF(
        model=model,
        processor=MagicMock(),
        inference_config=None,
        device=torch.device("cpu"),
    )

    # when
    _ = qwen.generate(inputs={"input_ids": torch.zeros((1, 5), dtype=torch.long)})

    # then
    assert (
        model.generate.call_args.kwargs["do_sample"]
        == INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE
    )


def test_prompt_applies_qwen3_8_defaults(monkeypatch) -> None:
    # Qwen35HF.prompt hardcodes its own defaults - the override must resolve
    # the qwen3_8 env-derived ones on this entry point too.
    qwen = Qwen38HF(
        model=MagicMock(),
        processor=MagicMock(),
        inference_config=None,
        device=torch.device("cpu"),
    )
    captured = {}

    def fake_pre_process_generation(**kwargs):
        return {"input_ids": torch.zeros((1, 5), dtype=torch.long)}

    def fake_generate(inputs, max_new_tokens=None, do_sample=None, **kwargs):
        captured["max_new_tokens"] = max_new_tokens
        captured["do_sample"] = do_sample
        return torch.zeros((1, 3), dtype=torch.long)

    monkeypatch.setattr(qwen, "pre_process_generation", fake_pre_process_generation)
    monkeypatch.setattr(qwen, "generate", fake_generate)
    monkeypatch.setattr(qwen, "post_process_generation", lambda **kwargs: ["ok"])

    # when
    result = qwen.prompt(images=MagicMock())

    # then
    assert result == ["ok"]
    assert captured["max_new_tokens"] == INFERENCE_MODELS_QWEN3_8_DEFAULT_MAX_NEW_TOKENS
    assert captured["do_sample"] == INFERENCE_MODELS_QWEN3_8_DEFAULT_DO_SAMPLE


def test_from_pretrained_rejects_old_transformers(monkeypatch) -> None:
    from inference_models.errors import EnvironmentConfigurationError
    from inference_models.models.qwen3_8 import qwen3_8_hf

    monkeypatch.setattr(qwen3_8_hf.transformers, "__version__", "5.5.0")

    with pytest.raises(EnvironmentConfigurationError, match="5.8.0"):
        Qwen38HF.from_pretrained("/nonexistent")


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
