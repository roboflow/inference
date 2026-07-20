from unittest.mock import MagicMock

import numpy as np
import torch

from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


def _model_with_processor() -> Cosmos3EdgeReasoner:
    model = MagicMock()
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
    assert (
        conversation[1]["content"][1]["text"] == "Describe what's in this image."
    )


def test_pre_process_generation_video_path_passes_frames_as_video() -> None:
    reasoner = _model_with_processor()
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    reasoner.pre_process_generation(images=frames, as_video=True)

    conversation = reasoner._processor.apply_chat_template.call_args.args[0]
    assert conversation[1]["content"][0]["type"] == "video"
    assert "videos" in reasoner._processor.call_args.kwargs
    assert len(reasoner._processor.call_args.kwargs["videos"][0]) == 4


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
