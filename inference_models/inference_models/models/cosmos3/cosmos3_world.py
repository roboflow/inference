from dataclasses import dataclass, field
from threading import Lock
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import ModelInputError
from inference_models.models.cosmos3.runtime_loading import load_runtime_from_package

RUNTIME_MODULE_FILE = "cosmos3_generator_runtime.py"
RUNTIME_CLASS_NAME = "Cosmos3GeneratorRuntime"
SESSION_KEY = "cosmos3_rollout_session"

DEFAULT_NUM_FRAMES = 121
DEFAULT_FPS = 24
DEFAULT_RESOLUTION = 480


@dataclass
class Cosmos3ActionTrajectory:
    """A robot action trajectory - consumed by forward dynamics, produced by
    inverse dynamics."""

    actions: np.ndarray  # (T, action_dim)
    action_space: str
    fps: float
    metadata: dict = field(default_factory=dict)


@dataclass
class Cosmos3Rollout:
    """Result of a generation / dynamics call.

    `frames` are BGR (repo-wide convention). `state_dict` is opaque live
    session state - thread it back into the next `forward_dynamics()` call,
    never introspect or serialize it.
    """

    frames: List[np.ndarray]
    fps: float
    resolution: Tuple[int, int]
    actions: Optional[Cosmos3ActionTrajectory] = None
    state_dict: Optional[dict] = None


class Cosmos3EdgeWorldModel:
    """NVIDIA Cosmos 3 Edge generator tower.

    Modes exposed: image-to-video (`generate_video`), forward dynamics
    (`start_rollout` + `forward_dynamics`), inverse dynamics
    (`inverse_dynamics`). The policy mode (step-wise robot control) is
    deliberately not exposed yet.
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        **kwargs,
    ) -> "Cosmos3EdgeWorldModel":
        runtime_class = load_runtime_from_package(
            model_name_or_path=model_name_or_path,
            runtime_module_file=RUNTIME_MODULE_FILE,
            runtime_class_name=RUNTIME_CLASS_NAME,
        )
        runtime = runtime_class.load(model_name_or_path, device=device)
        return cls(runtime=runtime, device=device)

    def __init__(self, runtime, device: torch.device):
        self._runtime = runtime
        self._device = device
        self._lock = Lock()

    def generate_video(
        self,
        images: Union[np.ndarray, List[np.ndarray]],
        prompt: Optional[str] = None,
        num_frames: int = DEFAULT_NUM_FRAMES,
        fps: int = DEFAULT_FPS,
        resolution: int = DEFAULT_RESOLUTION,
        seed: Optional[int] = None,
        input_color_format: ColorFormat = None,
        **kwargs,
    ) -> Cosmos3Rollout:
        conditioning = _as_rgb_frames(images, input_color_format=input_color_format)
        with self._lock:
            generated = self._runtime.generate_video(
                images=conditioning,
                prompt=prompt,
                num_frames=num_frames,
                fps=fps,
                resolution=resolution,
                seed=seed,
            )
        frames = [_rgb_to_bgr(frame) for frame in generated]
        return Cosmos3Rollout(
            frames=frames,
            fps=float(fps),
            resolution=_frames_resolution(frames),
        )

    def start_rollout(
        self,
        frames: List[np.ndarray],
        input_color_format: ColorFormat = None,
    ) -> dict:
        context = _as_rgb_frames(frames, input_color_format=input_color_format)
        if not context:
            raise ModelInputError(
                message="start_rollout() requires at least one context frame.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        with self._lock:
            session = self._runtime.encode_context(frames=context)
        return {SESSION_KEY: session}

    def forward_dynamics(
        self,
        actions: Cosmos3ActionTrajectory,
        state_dict: dict,
        num_frames: Optional[int] = None,
        **kwargs,
    ) -> Cosmos3Rollout:
        session = self._resolve_session(state_dict=state_dict)
        if num_frames is None:
            num_frames = len(actions.actions)
        with self._lock:
            generated, updated_session = self._runtime.rollout(
                session=session,
                actions=actions.actions,
                num_frames=num_frames,
            )
        frames = [_rgb_to_bgr(frame) for frame in generated]
        return Cosmos3Rollout(
            frames=frames,
            fps=actions.fps,
            resolution=_frames_resolution(frames),
            state_dict={SESSION_KEY: updated_session},
        )

    def inverse_dynamics(
        self,
        frames: List[np.ndarray],
        input_color_format: ColorFormat = None,
        **kwargs,
    ) -> Cosmos3ActionTrajectory:
        observation = _as_rgb_frames(frames, input_color_format=input_color_format)
        if not observation:
            raise ModelInputError(
                message="inverse_dynamics() requires at least one observation frame.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        with self._lock:
            result = self._runtime.infer_actions(frames=observation)
        return Cosmos3ActionTrajectory(
            actions=np.asarray(result["actions"]),
            action_space=result["action_space"],
            fps=float(result["fps"]),
            metadata=result.get("metadata", {}),
        )

    def _resolve_session(self, state_dict: dict):
        # A missing / foreign state_dict must fail loudly - silently creating
        # an empty session would produce plausible-looking garbage rollouts.
        if not isinstance(state_dict, dict) or SESSION_KEY not in state_dict:
            raise ModelInputError(
                message="forward_dynamics() requires the state_dict returned by "
                "start_rollout() (or by a previous forward_dynamics() call). "
                "Call start_rollout() with context frames first.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        return state_dict[SESSION_KEY]


def _as_rgb_frames(
    images: Union[np.ndarray, List[np.ndarray]],
    input_color_format: ColorFormat,
) -> List[np.ndarray]:
    if isinstance(images, np.ndarray):
        images = [images]
    frames = []
    for frame in images:
        if not isinstance(frame, np.ndarray):
            raise ModelInputError(
                message="Cosmos 3 generator inputs must be numpy arrays "
                f"(got {type(frame).__name__}).",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        if input_color_format != "rgb":
            frame = frame[:, :, ::-1]
        frames.append(np.ascontiguousarray(frame))
    return frames


def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(frame)[:, :, ::-1])


def _frames_resolution(frames: List[np.ndarray]) -> Tuple[int, int]:
    if not frames:
        return (0, 0)
    height, width = frames[0].shape[:2]
    return (width, height)
