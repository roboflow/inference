"""Qwen-Image-Edit HF backend.

Wraps the Qwen/Qwen-Image-Edit pipeline so it satisfies the
`from_pretrained(model_name_or_path)` contract expected by `inference_models.AutoModel`.

The model takes a source image and a text editing instruction and returns an
edited PIL Image. It is loaded via `diffusers.QwenImageEditPipeline`.

Optionally, the lightx2v **Qwen-Image-Lightning** LoRA can be loaded. That LoRA
is a step-distillation adapter: with the matching FlowMatch scheduler it produces
results in ~4 diffusion steps (guidance disabled), which — together with
sequential CPU offload, attention/VAE slicing and input downscaling — makes the
otherwise very heavy model usable on consumer GPUs.
"""

import logging
import math
import os
from threading import Lock
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_GUIDANCE_SCALE,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_NUM_INFERENCE_STEPS,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_GUIDANCE_SCALE,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_MAX_MEGAPIXELS,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_NUM_INFERENCE_STEPS,
    INFERENCE_MODELS_QWEN_IMAGE_EDIT_USE_LIGHTNING_LORA,
)

logger = logging.getLogger(__name__)

MODEL_ID = os.getenv("QWEN_EDIT_MODEL_ID", "Qwen/Qwen-Image-Edit")
LORA_REPO = os.getenv("QWEN_LIGHTNING_LORA_REPO", "lightx2v/Qwen-Image-Lightning")
LORA_FILE = os.getenv(
    "QWEN_LIGHTNING_LORA_FILE",
    "Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors",
)

_CPU_OFFLOAD_ENV_VAR = "INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD"

# The Lightning step-distillation LoRA only produces good results with this
# specific FlowMatchEuler scheduler configuration (matches lightx2v's recipe).
_LIGHTNING_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


class QwenImageEditHF:
    """Thin wrapper around the Qwen-Image-Edit pipeline."""

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = MODEL_ID,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        use_lightning_lora: bool = INFERENCE_MODELS_QWEN_IMAGE_EDIT_USE_LIGHTNING_LORA,
        lora_repo: str = LORA_REPO,
        lora_file: str = LORA_FILE,
        **kwargs,
    ) -> "QwenImageEditHF":
        from diffusers import QwenImageEditPipeline

        torch_dtype = cls.default_dtype if device.type == "cuda" else torch.float32

        pipeline_kwargs = {
            "local_files_only": local_files_only,
            "torch_dtype": torch_dtype,
        }
        if use_lightning_lora:
            # Lightning needs its own scheduler, and the transformer must be
            # materialized (low_cpu_mem_usage=False) so sequential CPU offload
            # and LoRA loading don't trip over meta tensors.
            pipeline_kwargs["low_cpu_mem_usage"] = False
            pipeline_kwargs["scheduler"] = _build_lightning_scheduler()
            pipeline_kwargs["transformer"] = _load_transformer(
                model_name_or_path=model_name_or_path,
                torch_dtype=torch_dtype,
                local_files_only=local_files_only,
            )

        pipe = QwenImageEditPipeline.from_pretrained(
            model_name_or_path,
            **pipeline_kwargs,
        )

        lightning_lora_applied = False
        if use_lightning_lora:
            lightning_lora_applied = _load_lightning_lora(
                pipe=pipe,
                lora_repo=lora_repo,
                lora_file=lora_file,
                local_files_only=local_files_only,
            )

        _place_on_device(
            pipe=pipe,
            device=device,
            use_lightning_lora=lightning_lora_applied,
        )

        return cls(
            pipeline=pipe,
            device=device,
            lightning_lora_applied=lightning_lora_applied,
        )

    def __init__(
        self,
        pipeline,
        device: torch.device,
        lightning_lora_applied: bool = False,
    ):
        self._pipeline = pipeline
        self._device = device
        self._lightning_lora_applied = lightning_lora_applied
        self._lock = Lock()

    @property
    def lightning_lora_applied(self) -> bool:
        return self._lightning_lora_applied

    def edit(
        self,
        image: Union[np.ndarray, Image.Image],
        prompt: str,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        negative_prompt: str = " ",
        seed: Optional[int] = None,
        scale_megapixels: Optional[float] = None,
        **kwargs,
    ) -> Image.Image:
        pil_image = _to_pil(image)
        target_mp = self._resolve_scale_megapixels(scale_megapixels)
        if target_mp is not None:
            pil_image = _scale_to_megapixels(pil_image, target_mp)

        steps, cfg_scale = self._resolve_generation_defaults(
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )

        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)

        logger.info(
            "Running Qwen-Image-Edit at %dx%d (%d steps, cfg %s, lightning=%s).",
            pil_image.width,
            pil_image.height,
            steps,
            cfg_scale,
            self._lightning_lora_applied,
        )

        with self._lock, torch.inference_mode():
            result = self._pipeline(
                image=pil_image,
                prompt=prompt,
                num_inference_steps=steps,
                true_cfg_scale=cfg_scale,
                negative_prompt=negative_prompt,
                generator=generator,
                **kwargs,
            )
            if self._device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

        return result.images[0]

    def _resolve_generation_defaults(
        self,
        num_inference_steps: Optional[int],
        guidance_scale: Optional[float],
    ) -> tuple:
        """Pick step/guidance defaults, honouring the Lightning recipe when active.

        Explicit caller-provided values always win; ``None`` means "auto".
        """
        if self._lightning_lora_applied:
            default_steps = (
                INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_NUM_INFERENCE_STEPS
            )
            default_cfg = INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_GUIDANCE_SCALE
        else:
            default_steps = INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_NUM_INFERENCE_STEPS
            default_cfg = INFERENCE_MODELS_QWEN_IMAGE_EDIT_DEFAULT_GUIDANCE_SCALE
        steps = (
            num_inference_steps if num_inference_steps is not None else default_steps
        )
        cfg_scale = guidance_scale if guidance_scale is not None else default_cfg
        return steps, cfg_scale

    def _resolve_scale_megapixels(
        self, scale_megapixels: Optional[float]
    ) -> Optional[float]:
        """Resolve the input downscale cap. ``None`` means "auto".

        With the Lightning LoRA we apply the configured cap by default (diffusion
        VRAM scales with pixel count); for the full model we leave inputs as-is.
        """
        if scale_megapixels is not None:
            return scale_megapixels
        if self._lightning_lora_applied:
            return INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_MAX_MEGAPIXELS
        return None


def _build_lightning_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler

    return FlowMatchEulerDiscreteScheduler.from_config(_LIGHTNING_SCHEDULER_CONFIG)


def _load_transformer(
    model_name_or_path: str,
    torch_dtype: torch.dtype,
    local_files_only: bool,
):
    from diffusers.models import QwenImageTransformer2DModel

    return QwenImageTransformer2DModel.from_pretrained(
        model_name_or_path,
        subfolder="transformer",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        local_files_only=local_files_only,
    )


def _load_lightning_lora(
    pipe,
    lora_repo: str,
    lora_file: str,
    local_files_only: bool,
) -> bool:
    """Load the Lightning LoRA as an active adapter. Returns True on success.

    The LoRA is intentionally *not* fused: keeping it as an adapter avoids
    materializing fused weights, which is incompatible with sequential CPU
    offload. Failures are non-fatal — we log and fall back to the base weights.
    """
    try:
        from huggingface_hub import hf_hub_download

        lora_path = hf_hub_download(
            repo_id=lora_repo,
            filename=lora_file,
            local_files_only=local_files_only,
        )
        pipe.load_lora_weights(os.path.dirname(lora_path), weight_name=lora_file)
        logger.info(
            "Loaded Qwen-Image-Lightning LoRA '%s/%s'; recommended generation: "
            "~%s steps with guidance %s.",
            lora_repo,
            lora_file,
            INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_NUM_INFERENCE_STEPS,
            INFERENCE_MODELS_QWEN_IMAGE_EDIT_LIGHTNING_GUIDANCE_SCALE,
        )
        return True
    except Exception:
        logger.exception(
            "Failed to load Lightning LoRA '%s/%s'; falling back to base "
            "Qwen-Image-Edit weights.",
            lora_repo,
            lora_file,
        )
        return False


def _place_on_device(pipe, device: torch.device, use_lightning_lora: bool) -> None:
    """Place the pipeline on the device using the configured offload strategy.

    The Lightning path defaults to sequential offload + slicing (what fits the
    base model on <=24GB cards). The env var, when explicitly set, always wins.
    """
    if device.type != "cuda":
        pipe.to(device)
        return

    if _CPU_OFFLOAD_ENV_VAR in os.environ:
        offload_mode = INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD
    elif use_lightning_lora:
        offload_mode = "sequential"
    else:
        offload_mode = INFERENCE_MODELS_QWEN_IMAGE_EDIT_CPU_OFFLOAD

    if offload_mode == "sequential":
        # Submodule-level offload: lowest VRAM, needed to fit the full base
        # model on <=24GB cards, at the cost of speed.
        pipe.enable_sequential_cpu_offload()
        _enable_slicing(pipe)
    elif offload_mode == "none":
        pipe.to(device)
    else:
        # Default: keep one sub-model on the GPU at a time.
        pipe.enable_model_cpu_offload()


def _enable_slicing(pipe) -> None:
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing(slice_size="auto")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()


def _scale_to_megapixels(pil_img: Image.Image, scale_mp: float) -> Image.Image:
    """Downscale (never upscale) so the image is at most ``scale_mp`` megapixels.

    Dimensions are snapped to multiples of 8 as required by the VAE.
    """
    w, h = pil_img.size
    mp = (w * h) / 1_000_000
    if mp <= scale_mp:
        return pil_img
    scale = (scale_mp / mp) ** 0.5
    new_w = max(8, (int(w * scale) // 8) * 8)
    new_h = max(8, (int(h * scale) // 8) * 8)
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _to_pil(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    # numpy: assume BGR (OpenCV convention used throughout inference)
    rgb = image[:, :, ::-1].copy() if image.ndim == 3 else image
    return Image.fromarray(rgb.astype(np.uint8))
