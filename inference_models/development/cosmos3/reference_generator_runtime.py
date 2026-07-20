"""Reference `cosmos3_generator_runtime.py` for the Cosmos 3 Edge world package.

This is the self-contained runtime module that ships INSIDE the
`cosmos-3-edge-world` model package (injected by pull_weights.py
--runtime-module). `Cosmos3EdgeWorldModel.from_pretrained` imports the
`Cosmos3GeneratorRuntime` class from it via `import_class_from_file`.

Backed by diffusers' `Cosmos3OmniPipeline` (diffusers>=0.39), so the only
runtime dependencies are torch + diffusers + transformers - NVIDIA's
cosmos-framework is not required for image-to-video.

Contract expected by Cosmos3EdgeWorldModel (all images RGB numpy arrays):
- load(package_dir, device) -> runtime
- generate_video(images, prompt, num_frames, fps, resolution, seed) -> [frames]
- encode_context(frames) -> session
- rollout(session, actions, num_frames) -> ([frames], session)
- infer_actions(frames) -> {actions, action_space, fps, metadata}

Forward/inverse dynamics ride the pipeline's action-conditioning path
(`CosmosActionCondition`): forward dynamics conditions each chunk on the last
generated frame of the previous chunk (carried in the session dict), inverse
dynamics infers the action sequence connecting the frames of an observation
video. `action_space` maps to the pipeline's embodiment `domain_name`
(e.g. "umi", "av", "droid_lerobot").
"""

from typing import List, Optional

import numpy as np
import torch


class Cosmos3GeneratorRuntime:

    @classmethod
    def load(cls, package_dir: str, device: torch.device) -> "Cosmos3GeneratorRuntime":
        from diffusers import Cosmos3OmniPipeline

        pipeline = Cosmos3OmniPipeline.from_pretrained(
            package_dir,
            torch_dtype=torch.bfloat16,
        )
        pipeline.to(device)
        return cls(pipeline=pipeline, device=device)

    def __init__(self, pipeline, device: torch.device):
        self._pipeline = pipeline
        self._device = device

    def generate_video(
        self,
        images: List[np.ndarray],
        prompt: Optional[str],
        num_frames: int,
        fps: int,
        resolution: int,
        seed: Optional[int],
        num_inference_steps: int = 35,
    ) -> List[np.ndarray]:
        from PIL import Image

        conditioning = Image.fromarray(images[0])
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        height, width = _resolve_resolution(resolution)
        result = self._pipeline(
            prompt=prompt or "",
            image=conditioning,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=float(fps),
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="np",
            enable_safety_check=False,
        )
        return [frame for frame in _to_uint8(result.video)]

    def encode_context(
        self,
        frames: List[np.ndarray],
        prompt: Optional[str] = None,
        view_point: str = "ego_view",
        resolution_tier: int = 256,
        **kwargs,
    ) -> dict:
        # The session carries the frame each subsequent chunk conditions on,
        # plus the generation context that stays fixed across the rollout.
        return {
            "conditioning_frame": np.asarray(frames[-1]),
            "prompt": prompt,
            "view_point": view_point,
            "resolution_tier": resolution_tier,
        }

    def rollout(
        self,
        session: dict,
        actions: np.ndarray,
        num_frames: int,
        action_space: str = "umi",
        seed: Optional[int] = None,
        num_inference_steps: int = 35,
        **kwargs,
    ):
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            CosmosActionCondition,
        )
        from PIL import Image

        condition = CosmosActionCondition(
            mode="forward_dynamics",
            chunk_size=len(actions),
            domain_name=action_space,
            resolution_tier=session["resolution_tier"],
            raw_actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            image=Image.fromarray(session["conditioning_frame"]),
            view_point=session["view_point"],
        )
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self._device).manual_seed(seed)
        result = self._pipeline(
            prompt=session.get("prompt") or "",
            action=condition,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="np",
            enable_safety_check=False,
        )
        video = _to_uint8(result.video)
        updated_session = dict(session)
        updated_session["conditioning_frame"] = video[-1]
        return [frame for frame in video], updated_session

    def infer_actions(
        self,
        frames: List[np.ndarray],
        action_space: str = "umi",
        view_point: str = "ego_view",
        resolution_tier: int = 256,
        fps: float = 20.0,
        num_inference_steps: int = 35,
        **kwargs,
    ) -> dict:
        from diffusers.pipelines.cosmos.pipeline_cosmos3_omni import (
            CosmosActionCondition,
        )

        condition = CosmosActionCondition(
            mode="inverse_dynamics",
            chunk_size=max(1, len(frames) - 1),
            domain_name=action_space,
            resolution_tier=resolution_tier,
            video=[np.asarray(frame) for frame in frames],
            view_point=view_point,
        )
        result = self._pipeline(
            prompt="",
            action=condition,
            num_inference_steps=num_inference_steps,
            output_type="np",
            enable_safety_check=False,
        )
        chunks = [chunk.float().cpu().numpy() for chunk in (result.action or [])]
        actions = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0))
        return {
            "actions": actions,
            "action_space": action_space,
            "fps": fps,
            "metadata": {"view_point": view_point, "num_chunks": len(chunks)},
        }


def _to_uint8(video) -> np.ndarray:
    video = np.asarray(video)
    if video.dtype != np.uint8:
        video = (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)
    return video


def _resolve_resolution(resolution: int):
    # 16:9 at the requested height, aligned to the VAE's spatial stride.
    height = max(16, (resolution // 16) * 16)
    width = max(16, ((resolution * 16 // 9) // 16) * 16)
    return height, width
