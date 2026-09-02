from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.models.cosmos3.cosmos3_reasoner_inference_models import (
    InferenceModelsCosmos3ReasonerAdapter,
)


def _adapter() -> InferenceModelsCosmos3ReasonerAdapter:
    adapter = InferenceModelsCosmos3ReasonerAdapter.__new__(
        InferenceModelsCosmos3ReasonerAdapter
    )
    adapter._model = MagicMock()
    adapter._model.pre_process_generation.return_value = {"input_ids": "encoded"}
    return adapter


def test_preprocess_single_image_is_an_image_prompt() -> None:
    adapter = _adapter()
    image = np.zeros((6, 9, 3), dtype=np.uint8)

    inputs, metadata = adapter.preprocess(image, prompt="What happens?")

    assert inputs == {"input_ids": "encoded"}
    assert metadata["image_dims"] == (9, 6)
    call = adapter._model.pre_process_generation.call_args
    assert call.args[1] == "What happens?"
    assert "as_video" not in call.kwargs


def test_preprocess_list_of_images_is_one_clip_at_video_fps() -> None:
    adapter = _adapter()
    frames = [np.zeros((6, 9, 3), dtype=np.uint8) for _ in range(3)]

    inputs, metadata = adapter.preprocess(frames, prompt="", video_fps=4.0)

    assert inputs == {"input_ids": "encoded"}
    assert metadata["image_dims"] == (9, 6)
    call = adapter._model.pre_process_generation.call_args
    assert len(call.args[0]) == 3
    assert call.kwargs["as_video"] is True
    assert call.kwargs["video_fps"] == 4.0


def test_preprocess_list_of_images_requires_video_fps() -> None:
    adapter = _adapter()
    frames = [np.zeros((6, 9, 3), dtype=np.uint8)]

    with pytest.raises(ValueError):
        adapter.preprocess(frames, prompt="")


def test_preprocess_rejects_an_empty_clip() -> None:
    adapter = _adapter()

    with pytest.raises(ValueError):
        adapter.preprocess([], prompt="", video_fps=4.0)
