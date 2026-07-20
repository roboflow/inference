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

Forward/inverse dynamics need the action-conditioning path
(`Cosmos3OmniPipeline`'s `action=` input); wiring for it is not implemented in
this reference module yet and the corresponding methods raise.
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
        video = np.asarray(result.video)
        if video.dtype != np.uint8:
            video = (np.clip(video, 0.0, 1.0) * 255).astype(np.uint8)
        return [frame for frame in video]

    def encode_context(self, frames: List[np.ndarray]):
        raise NotImplementedError(
            "Forward dynamics is not wired in the reference runtime yet - it "
            "requires the action-conditioning inputs of Cosmos3OmniPipeline."
        )

    def rollout(self, session, actions: np.ndarray, num_frames: int):
        raise NotImplementedError(
            "Forward dynamics is not wired in the reference runtime yet - it "
            "requires the action-conditioning inputs of Cosmos3OmniPipeline."
        )

    def infer_actions(self, frames: List[np.ndarray]) -> dict:
        raise NotImplementedError(
            "Inverse dynamics is not wired in the reference runtime yet - it "
            "requires the action decoding path of Cosmos3OmniPipeline."
        )


def _resolve_resolution(resolution: int):
    # 16:9 at the requested height, aligned to the VAE's spatial stride.
    height = max(16, (resolution // 16) * 16)
    width = max(16, ((resolution * 16 // 9) // 16) * 16)
    return height, width
