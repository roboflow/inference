from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import ModelInputError
from inference_models.models.cosmos3.cosmos3_world import (
    SESSION_KEY,
    Cosmos3ActionTrajectory,
    Cosmos3EdgeWorldModel,
)


def _world_model() -> Cosmos3EdgeWorldModel:
    return Cosmos3EdgeWorldModel(runtime=MagicMock(), device=torch.device("cpu"))


def _rgb_frames(n: int) -> list:
    return [np.full((4, 6, 3), 100, dtype=np.uint8) for _ in range(n)]


def test_generate_video_converts_output_to_bgr_and_reports_resolution() -> None:
    model = _world_model()
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 0] = 255  # pure red in RGB
    model._runtime.generate_video.return_value = [rgb, rgb]

    rollout = model.generate_video(
        images=np.zeros((4, 6, 3), dtype=np.uint8), num_frames=2, fps=24
    )

    assert len(rollout.frames) == 2
    assert rollout.frames[0][0, 0].tolist() == [0, 0, 255]  # red lands in BGR[2]
    assert rollout.resolution == (6, 4)
    assert rollout.fps == 24.0
    assert rollout.state_dict is None


def test_generate_video_converts_bgr_input_to_rgb_for_runtime() -> None:
    model = _world_model()
    model._runtime.generate_video.return_value = []
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR

    model.generate_video(images=bgr)

    sent = model._runtime.generate_video.call_args.kwargs["images"][0]
    assert sent[0, 0].tolist() == [0, 0, 255]  # blue lands in RGB[2]


def test_forward_dynamics_without_session_raises() -> None:
    model = _world_model()
    trajectory = Cosmos3ActionTrajectory(
        actions=np.zeros((5, 7)), action_space="droid_joint_velocity", fps=15.0
    )

    with pytest.raises(ModelInputError):
        model.forward_dynamics(actions=trajectory, state_dict={})
    with pytest.raises(ModelInputError):
        model.forward_dynamics(actions=trajectory, state_dict=None)


def test_forward_dynamics_threads_session_state() -> None:
    model = _world_model()
    model._runtime.encode_context.return_value = "session-0"
    model._runtime.rollout.return_value = (_rgb_frames(5), "session-1")
    trajectory = Cosmos3ActionTrajectory(
        actions=np.zeros((5, 7)), action_space="droid_joint_velocity", fps=15.0
    )

    state = model.start_rollout(frames=_rgb_frames(2))
    rollout = model.forward_dynamics(actions=trajectory, state_dict=state)

    assert state == {SESSION_KEY: "session-0"}
    assert model._runtime.rollout.call_args.kwargs["session"] == "session-0"
    assert model._runtime.rollout.call_args.kwargs["num_frames"] == 5
    assert rollout.state_dict == {SESSION_KEY: "session-1"}
    assert len(rollout.frames) == 5
    assert rollout.fps == 15.0


def test_start_rollout_requires_frames() -> None:
    model = _world_model()

    with pytest.raises(ModelInputError):
        model.start_rollout(frames=[])


def test_inverse_dynamics_wraps_runtime_result() -> None:
    model = _world_model()
    model._runtime.infer_actions.return_value = {
        "actions": [[0.1] * 7] * 3,
        "action_space": "droid_joint_velocity",
        "fps": 15,
        "metadata": {"source": "test"},
    }

    trajectory = model.inverse_dynamics(frames=_rgb_frames(3))

    assert trajectory.actions.shape == (3, 7)
    assert trajectory.action_space == "droid_joint_velocity"
    assert trajectory.fps == 15.0
    assert trajectory.metadata == {"source": "test"}
