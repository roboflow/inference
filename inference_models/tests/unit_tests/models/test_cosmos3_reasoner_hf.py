import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.configuration import (
    INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.cosmos3 import cosmos3_reasoner_hf as reasoner_module
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


def _model_with_processor() -> Cosmos3EdgeReasoner:
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    processor = MagicMock()
    processor.apply_chat_template.return_value = "templated"
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
    }
    return Cosmos3EdgeReasoner(
        model=model, processor=processor, device=torch.device("cpu")
    )


def test_generate_returns_only_new_tokens() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 21, 22]])

    result = reasoner.generate(inputs={"input_ids": torch.tensor([[1, 2]])})

    assert result.tolist() == [[21, 22]]


def test_pre_process_generation_builds_system_and_user_turns() -> None:
    reasoner = _model_with_processor()

    inputs = reasoner.pre_process_generation(
        images=np.zeros((8, 8, 3), dtype=np.uint8),
        prompt="What is happening?<system_prompt>Be terse.",
    )

    conversation = reasoner._processor.apply_chat_template.call_args.args[0]
    assert conversation[0]["role"] == "system"
    assert conversation[0]["content"][0]["text"] == "Be terse."
    assert conversation[1]["content"][1]["text"] == "What is happening?"
    assert "input_ids" in inputs and "pixel_values" in inputs


def test_pre_process_generation_uses_defaults_without_prompt() -> None:
    reasoner = _model_with_processor()

    reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    conversation = reasoner._processor.apply_chat_template.call_args.args[0]
    assert conversation[0]["content"][0]["text"] == reasoner.default_system_prompt
    assert conversation[1]["content"][1]["text"] == "Describe what's in this image."


def test_pre_process_generation_video_path_passes_frames_as_video() -> None:
    reasoner = _model_with_processor()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    reasoner.pre_process_generation(images=frames, as_video=True)

    conversation = reasoner._processor.apply_chat_template.call_args.args[0]
    assert conversation[1]["content"][0]["type"] == "video"
    assert "videos" in reasoner._processor.call_args.kwargs
    assert len(reasoner._processor.call_args.kwargs["videos"][0]) == 4


def test_pre_process_generation_casts_floating_point_inputs_to_model_dtype() -> None:
    reasoner = _model_with_processor()

    inputs = reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert inputs["input_ids"].dtype == torch.int64
    assert inputs["pixel_values"].dtype == torch.bfloat16


def test_generate_uses_default_max_new_tokens_when_none_is_given() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 9]])

    reasoner.generate(inputs={"input_ids": torch.tensor([[1, 2]])}, max_new_tokens=None)

    assert reasoner._model.generate.call_args.kwargs["max_new_tokens"] == (
        INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS
    )


def test_post_process_generation_strips_thinking_block_by_default() -> None:
    reasoner = _model_with_processor()
    reasoner._processor.batch_decode.return_value = [
        "The image is blue. So the answer is blue.</think>  blue"
    ]

    result = reasoner.post_process_generation(generated_ids=torch.tensor([[1]]))

    assert result == ["blue"]


def test_post_process_generation_returns_thinking_when_requested() -> None:
    reasoner = _model_with_processor()
    reasoner._processor.batch_decode.return_value = [
        "The image is blue. So the answer is blue.</think>  blue"
    ]

    result = reasoner.post_process_generation(
        generated_ids=torch.tensor([[1]]), return_thinking=True
    )

    assert result == [
        {
            "thinking": "The image is blue. So the answer is blue.",
            "answer": "blue",
        }
    ]


def test_post_process_generation_handles_truncated_thinking() -> None:
    reasoner = _model_with_processor()
    reasoner._processor.batch_decode.return_value = ["Okay, the user wants"]

    result = reasoner.post_process_generation(
        generated_ids=torch.tensor([[1]]), return_thinking=True
    )

    assert result == [{"thinking": "Okay, the user wants", "answer": ""}]


def test_post_process_generation_strips_assistant_prefix() -> None:
    reasoner = _model_with_processor()
    reasoner._processor.batch_decode.return_value = ["assistant\nThe box falls.  "]

    result = reasoner.post_process_generation(generated_ids=torch.tensor([[1]]))

    assert result == ["The box falls."]


def test_prompt_video_returns_single_string() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
    reasoner._processor.batch_decode.return_value = ["a robot arm"]

    result = reasoner.prompt_video(
        frames=[np.zeros((8, 8, 3), dtype=np.uint8)] * 2,
        prompt="What will happen next?",
    )

    assert result == "a robot arm"


def _fake_loaded_model() -> MagicMock:
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    return model


def test_from_pretrained_loads_a_roboflow_fine_tune_from_its_adapter(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (tmp_path / "base").mkdir()
    base_model = MagicMock()
    merged = _fake_loaded_model()
    peft_model = MagicMock()
    peft_model.merge_and_unload.return_value = merged
    load_base = MagicMock(return_value=base_model)
    load_adapter = MagicMock(return_value=peft_model)
    load_processor = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(reasoner_module, "_require_cosmos3_transformers", lambda: None)
    monkeypatch.setattr(
        reasoner_module.AutoModelForImageTextToText, "from_pretrained", load_base
    )
    monkeypatch.setattr(reasoner_module.PeftModel, "from_pretrained", load_adapter)
    monkeypatch.setattr(
        reasoner_module.AutoProcessor, "from_pretrained", load_processor
    )

    reasoner = Cosmos3EdgeReasoner.from_pretrained(
        str(tmp_path), device=torch.device("cpu")
    )

    # The base checkpoint comes from base/, the adapter from the package root, and
    # the merged model is what serves.
    assert load_base.call_args.args[0] == str(tmp_path / "base")
    assert load_adapter.call_args.args == (base_model, str(tmp_path))
    assert load_processor.call_args.args[0] == str(tmp_path / "base")
    assert reasoner._model is merged
    # Fine-tunes are trained with the think block empty, so they answer directly.
    assert reasoner._enable_thinking is False
    assert reasoner.default_system_prompt == reasoner_module.FINE_TUNE_SYSTEM_PROMPT


def test_from_pretrained_refuses_a_fine_tune_with_class_tokens(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"trainable_token_indices": {"embed_tokens": [151669]}})
    )
    monkeypatch.setattr(reasoner_module, "_require_cosmos3_transformers", lambda: None)

    with pytest.raises(NotImplementedError, match="video fine-tune"):
        Cosmos3EdgeReasoner.from_pretrained(str(tmp_path), device=torch.device("cpu"))


def test_require_cosmos3_transformers_names_the_floor(monkeypatch) -> None:
    monkeypatch.setattr(reasoner_module.transformers, "__version__", "5.5.0")

    with pytest.raises(RuntimeError, match="transformers>=5.15.0"):
        reasoner_module._require_cosmos3_transformers()


def test_post_process_generation_returns_the_answer_as_answer_when_thinking_is_off() -> (
    None
):
    reasoner = _model_with_processor()
    reasoner._enable_thinking = False
    reasoner._processor.batch_decode.return_value = ["Beer cans on production line"]

    with_thinking = reasoner.post_process_generation(
        torch.tensor([[1]]), return_thinking=True
    )
    plain = reasoner.post_process_generation(torch.tensor([[1]]))

    assert with_thinking == [{"thinking": "", "answer": "Beer cans on production line"}]
    assert plain == ["Beer cans on production line"]


def test_pre_process_generation_leaves_thinking_on_for_the_base_model() -> None:
    reasoner = _model_with_processor()

    reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert (
        "enable_thinking"
        not in reasoner._processor.apply_chat_template.call_args.kwargs
    )


def test_pre_process_generation_turns_thinking_off_for_a_fine_tune() -> None:
    reasoner = _model_with_processor()
    reasoner._enable_thinking = False

    reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert (
        reasoner._processor.apply_chat_template.call_args.kwargs["enable_thinking"]
        is False
    )


def test_fine_tuned_reasoner_prompts_with_the_training_system_prompt() -> None:
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    processor = MagicMock()
    processor.apply_chat_template.return_value = "templated"
    processor.return_value = {"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64)}
    reasoner = Cosmos3EdgeReasoner(
        model=model, processor=processor, device=torch.device("cpu"), fine_tuned=True
    )

    reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    conversation = processor.apply_chat_template.call_args.args[0]
    assert (
        conversation[0]["content"][0]["text"] == reasoner_module.FINE_TUNE_SYSTEM_PROMPT
    )
    assert reasoner_module.FINE_TUNE_SYSTEM_PROMPT != reasoner_module.BASE_SYSTEM_PROMPT


VALID_INFERENCE_CONFIG = {
    "image_pre_processing": {"grayscale": {"enabled": True}},
    "network_input": {
        "training_input_size": {"height": 64, "width": 64},
        "dynamic_spatial_size_supported": True,
        "dynamic_spatial_size_mode": {"type": "any-size"},
        "color_mode": "rgb",
        "resize_mode": "stretch",
        "input_channels": 3,
    },
}
# What roboflow-train writes for a version without a resize: no training size, any-size model.
TRAINER_NULL_INFERENCE_CONFIG = {
    "image_pre_processing": None,
    "network_input": {
        "training_input_size": None,
        "dynamic_spatial_size_supported": True,
        "dynamic_spatial_size_mode": {"type": "any-size"},
        "color_mode": "rgb",
        "resize_mode": "stretch",
        "padding_value": None,
        "input_channels": 3,
        "scaling_factor": None,
        "normalization": None,
    },
}


def test_load_inference_config_parses_a_valid_file(tmp_path) -> None:
    (tmp_path / "inference_config.json").write_text(json.dumps(VALID_INFERENCE_CONFIG))

    config = reasoner_module._load_inference_config(str(tmp_path))

    assert config is not None
    assert config.network_input.training_input_size.height == 64


def test_load_inference_config_parses_the_trainers_any_size_config(tmp_path) -> None:
    (tmp_path / "inference_config.json").write_text(
        json.dumps(TRAINER_NULL_INFERENCE_CONFIG)
    )

    config = reasoner_module._load_inference_config(str(tmp_path))

    assert config is not None
    assert config.network_input.training_input_size is None
    assert reasoner_module._load_inference_config(str(tmp_path / "missing")) is None


def test_load_inference_config_rejects_a_config_the_schema_rejects(tmp_path) -> None:
    broken = json.loads(json.dumps(TRAINER_NULL_INFERENCE_CONFIG))
    broken["network_input"]["dynamic_spatial_size_supported"] = False
    (tmp_path / "inference_config.json").write_text(json.dumps(broken))

    with pytest.raises(CorruptedModelPackageError):
        reasoner_module._load_inference_config(str(tmp_path))


def test_pre_process_generation_applies_the_packages_preprocessing(monkeypatch) -> None:
    reasoner = _model_with_processor()
    reasoner._inference_config = reasoner_module.InferenceConfig.model_validate(
        VALID_INFERENCE_CONFIG
    )
    prepared = [torch.zeros((3, 8, 8))]
    pre_process = MagicMock(return_value=(prepared, [MagicMock()]))
    monkeypatch.setattr(
        reasoner_module,
        "pre_process_network_input_to_image_list",
        pre_process,
    )

    reasoner.pre_process_generation(images=np.zeros((8, 8, 3), dtype=np.uint8))

    assert (
        pre_process.call_args.kwargs["network_input"]
        is reasoner._inference_config.network_input
    )
    images_given = reasoner._processor.call_args.kwargs["images"]
    assert len(images_given) == 1 and images_given[0].shape == (3, 8, 8)
