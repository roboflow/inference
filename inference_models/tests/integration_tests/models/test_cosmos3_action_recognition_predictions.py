"""Integration tests for Cosmos 3 Edge action recognition on real weights.

The tests load from local package directories (the layouts produced by
`development/cosmos3/pull_weights.py`) pointed to by env vars, so they can run
before the packages reach the weights provider:

    COSMOS3_REASONER_PACKAGE_DIR=checkpoints/packages/cosmos-3-edge \
    COSMOS3_ACTION_RECOGNITION_PACKAGE_DIR=checkpoints/packages/my-fine-tune \
    python -m pytest \
      tests/integration_tests/models/test_cosmos3_action_recognition_predictions.py \
      -m slow

The reasoner package runs zero-shot. The action recognition package is a
fine-tune, which carries its own class list and its own sampling contract, so
the tests that need a declared vocabulary skip without it.

Once the packages are registered, conftest fixtures downloading the published
zips should replace the env-var indirection (matching the other model suites).
"""

import os
from typing import List

import numpy as np
import pytest
import torch

REASONER_PACKAGE_DIR = os.environ.get("COSMOS3_REASONER_PACKAGE_DIR")
FINE_TUNE_PACKAGE_DIR = os.environ.get("COSMOS3_ACTION_RECOGNITION_PACKAGE_DIR")
CUDA_AVAILABLE = torch.cuda.is_available()

requires_reasoner = pytest.mark.skipif(
    not REASONER_PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS3_REASONER_PACKAGE_DIR not set or CUDA unavailable",
)
requires_fine_tune = pytest.mark.skipif(
    not FINE_TUNE_PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS3_ACTION_RECOGNITION_PACKAGE_DIR not set or CUDA unavailable",
)


def _moving_square_frames(count: int = 16) -> List[np.ndarray]:
    """A white square crossing a black field, left to right."""
    frames = []
    for x in np.linspace(20, 240, num=count, dtype=int):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[96:144, x : x + 48] = 255
        frames.append(frame)
    return frames


def _load(package_dir: str):
    from inference_models.models.cosmos3.cosmos3_action_recognition import (
        Cosmos3EdgeActionRecognition,
    )

    return Cosmos3EdgeActionRecognition.from_pretrained(
        package_dir, device=torch.device("cuda")
    )


# One load per package, not per test: these weights are 4B parameters, and a
# second copy on the same device runs the GPU out of memory.
@pytest.fixture(scope="module")
def zero_shot_model():
    return _load(REASONER_PACKAGE_DIR)


@pytest.fixture(scope="module")
def fine_tune_model():
    return _load(FINE_TUNE_PACKAGE_DIR)


def _assert_usable_segments(segments, frame_count: int) -> None:
    assert isinstance(segments, list)
    assert segments
    for segment in segments:
        assert isinstance(segment.class_name, str) and segment.class_name.strip()
        assert isinstance(segment.start_frame_idx, int)
        assert isinstance(segment.end_frame_idx, int)
        assert 0 <= segment.start_frame_idx <= segment.end_frame_idx
        assert segment.end_frame_idx < frame_count


@pytest.mark.slow
@requires_reasoner
def test_zero_shot_localizes_moving_object(zero_shot_model) -> None:
    # given
    model = zero_shot_model
    frames = _moving_square_frames()

    # when
    # Base weights run zero-shot, which answers in its own words. Asserting a
    # requested phrase would test a vocabulary this mode ignores, so the gate
    # is that real weights return a usable range with a non-empty label.
    segments = model.infer(frames=frames, fps=8.0)

    # then
    _assert_usable_segments(segments, frame_count=len(frames))


@pytest.mark.slow
@requires_reasoner
def test_zero_shot_package_declares_no_trained_limits(zero_shot_model) -> None:
    # given / then
    # A model that never trained on a frame budget or a class list reports
    # neither. Handing zero-shot a fine-tune's limits broke it three ways.
    assert zero_shot_model.class_names is None
    assert zero_shot_model.video_sampling.max_frames is None
    assert zero_shot_model.video_sampling.sample_fps > 0


@pytest.mark.slow
@requires_reasoner
def test_zero_shot_accepts_cuda_tensor_frames(zero_shot_model) -> None:
    # given
    model = zero_shot_model
    frames = _moving_square_frames()
    # The tensor workflow block hands frames over as CHW RGB on the device.
    tensors = [torch.from_numpy(frame).permute(2, 0, 1).to("cuda") for frame in frames]

    # when
    segments = model.infer(frames=tensors, fps=8.0)

    # then
    _assert_usable_segments(segments, frame_count=len(tensors))


@pytest.mark.slow
@requires_reasoner
def test_infer_without_fps_raises(zero_shot_model) -> None:
    # given / then
    # The frame rate sets the timestamp on every frame the model reads, so
    # there is no safe default to fall back on.
    with pytest.raises(ValueError):
        zero_shot_model.infer(frames=_moving_square_frames(count=4))


@pytest.mark.slow
@requires_fine_tune
def test_fine_tune_declares_its_training_contract(fine_tune_model) -> None:
    # given / then
    assert fine_tune_model.class_names
    assert all(name.strip() for name in fine_tune_model.class_names)
    sampling = fine_tune_model.video_sampling
    assert sampling.max_frames is not None and sampling.max_frames > 0
    assert sampling.window_seconds > 0
    assert sampling.sample_fps > 0


@pytest.mark.slow
@requires_fine_tune
def test_fine_tune_answers_only_in_its_own_classes(fine_tune_model) -> None:
    # given
    model = fine_tune_model
    frames = _moving_square_frames(count=model.video_sampling.min_frames * 2)

    # when
    # Constrained decoding admits only tokens that continue a valid line, so
    # the answer parses and names a declared class every time.
    segments = model.infer(frames=frames, fps=model.video_sampling.sample_fps)

    # then
    _assert_usable_segments(segments, frame_count=len(frames))
    assert all(segment.class_name in model.class_names for segment in segments)


@pytest.mark.slow
@requires_fine_tune
def test_fine_tune_class_filter_narrows_the_vocabulary(fine_tune_model) -> None:
    # given
    model = fine_tune_model
    requested = model.class_names[:1]
    frames = _moving_square_frames(count=model.video_sampling.min_frames * 2)

    # when
    # The filter forms the prompt vocabulary, so the model has nothing else
    # to answer with.
    segments = model.infer(
        frames=frames,
        class_names=requested,
        fps=model.video_sampling.sample_fps,
    )

    # then
    assert all(segment.class_name in requested for segment in segments)
