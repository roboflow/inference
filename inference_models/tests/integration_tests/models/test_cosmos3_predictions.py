"""Integration tests for Cosmos 3 Edge (reasoner + world model) on real weights.

Both tests load from local package directories (the layouts produced by
`development/cosmos3/pull_weights.py`) pointed to by env vars, so they can run
before the packages are published to the weights provider:

    COSMOS3_REASONER_PACKAGE_DIR=checkpoints/packages/cosmos-3-edge \
    COSMOS3_WORLD_PACKAGE_DIR=checkpoints/packages/cosmos-3-edge-world \
    python -m pytest tests/integration_tests/models/test_cosmos3_predictions.py -m slow

Once the packages are registered, conftest fixtures downloading the published
zips should replace the env-var indirection (matching the other model suites).
"""

import os

import numpy as np
import pytest
import torch

REASONER_PACKAGE_DIR = os.environ.get("COSMOS3_REASONER_PACKAGE_DIR")
WORLD_PACKAGE_DIR = os.environ.get("COSMOS3_WORLD_PACKAGE_DIR")
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.slow
@pytest.mark.skipif(
    not REASONER_PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS3_REASONER_PACKAGE_DIR not set or CUDA unavailable",
)
def test_cosmos3_reasoner_answers_scene_question() -> None:
    from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
        Cosmos3EdgeReasoner,
    )

    model = Cosmos3EdgeReasoner.from_pretrained(
        REASONER_PACKAGE_DIR, device=torch.device("cuda")
    )
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, 160:, 2] = 255  # right half red (BGR)

    answers = model.prompt(
        images=image,
        prompt="Which side of the image is red? Answer with 'left' or 'right'.",
        max_new_tokens=64,
    )

    assert len(answers) == 1
    assert isinstance(answers[0], str)
    assert len(answers[0].strip()) > 0


@pytest.mark.slow
@pytest.mark.skipif(
    not WORLD_PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS3_WORLD_PACKAGE_DIR not set or CUDA unavailable",
)
def test_cosmos3_world_generates_video_from_image() -> None:
    from inference_models.models.cosmos3.cosmos3_world import Cosmos3EdgeWorldModel

    model = Cosmos3EdgeWorldModel.from_pretrained(
        WORLD_PACKAGE_DIR, device=torch.device("cuda")
    )
    conditioning = np.full((240, 416, 3), 128, dtype=np.uint8)

    rollout = model.generate_video(
        images=conditioning,
        prompt="static gray scene",
        num_frames=5,
        fps=24,
        resolution=240,
        seed=0,
        num_inference_steps=10,
    )

    assert len(rollout.frames) >= 1
    assert rollout.frames[0].ndim == 3
    assert rollout.frames[0].shape[2] == 3
    assert rollout.frames[0].dtype == np.uint8


@pytest.mark.slow
@pytest.mark.skipif(
    not WORLD_PACKAGE_DIR or not CUDA_AVAILABLE,
    reason="COSMOS3_WORLD_PACKAGE_DIR not set or CUDA unavailable",
)
def test_cosmos3_world_forward_dynamics_threads_session() -> None:
    from inference_models.models.cosmos3.cosmos3_world import (
        Cosmos3ActionTrajectory,
        Cosmos3EdgeWorldModel,
    )

    model = Cosmos3EdgeWorldModel.from_pretrained(
        WORLD_PACKAGE_DIR, device=torch.device("cuda")
    )
    first_frame = np.full((256, 256, 3), 128, dtype=np.uint8)
    actions = Cosmos3ActionTrajectory(
        actions=np.zeros((8, 10), dtype=np.float32),
        action_space="umi",
        fps=20.0,
    )

    state = model.start_rollout(frames=[first_frame], resolution_tier=256)
    rollout = model.forward_dynamics(
        actions=actions, state_dict=state, num_inference_steps=10
    )

    assert len(rollout.frames) >= 1
    assert rollout.state_dict is not None
    # the returned state must be usable for the next chunk
    second = model.forward_dynamics(
        actions=actions, state_dict=rollout.state_dict, num_inference_steps=10
    )
    assert len(second.frames) >= 1
