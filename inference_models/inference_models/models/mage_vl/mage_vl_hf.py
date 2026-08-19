import os
from threading import Lock
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_MAGE_VL_DEFAULT_CODEC_ENGINE,
    INFERENCE_MODELS_MAGE_VL_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_PIXELS,
    INFERENCE_MODELS_MAGE_VL_DEFAULT_TARGET_CANVAS,
    MAGE_VL_CODEC_ENGINES,
)
from inference_models.entities import ColorFormat
from inference_models.errors import (
    MissingDependencyError,
    ModelInputError,
    ModelRuntimeError,
)
from inference_models.models.common.model_packages import get_model_package_contents

VideoInput = Union[str, "os.PathLike[str]"]
ImageInput = Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]

CODEC_ENGINES = MAGE_VL_CODEC_ENGINES
# Mandated by the checkpoint's image processor (patch_size=16, merge_size=2). The
# `patch=14` seen in cv-preinfer's own config is an unrelated internal knob.
CODEC_PATCH_SIZE = 16
MAGE_VL_VIDEO_DEPS_HINT = (
    "Video prompting needs the `codec-video-prep` package (install it with "
    "`pip install --no-deps -r requirements/requirements.magevl.txt`, see that "
    "file for why), the `cv-preinfer` console script it ships on PATH (or "
    "CV_PREINFER_BIN pointing at it), and `ffmpeg`/`ffprobe` on PATH."
)


def _get_mage_vl_attn_implementation(device: torch.device) -> str:
    """Pick the attention kernel Mage-VL is documented to run with.

    Unlike the Qwen VL models in this package, eager attention is not a usable
    fallback here: the Mage-ViT encoder materializes the full attention matrix
    over the codec canvases and runs out of memory on smaller GPUs. Upstream
    recommends flash attention, falling back to SDPA.
    """
    if device is not None and device.type == "cuda" and is_flash_attn_2_available():
        try:
            import flash_attn  # noqa: F401

            major, _ = torch.cuda.get_device_capability(device=device)
            if major >= 8:
                return "flash_attention_2"
        except ImportError:
            pass
    return "sdpa"


def _register_config_class(config) -> None:
    """Make the checkpoint's config class known to transformers' auto-classes.

    The checkpoint's own `from_pretrained` drops `trust_remote_code` before it
    delegates to `AutoTokenizer`, which then resolves the config with the flag
    unset. Transformers only prompts (`input()`, 15s SIGALRM, then a hard failure)
    when it finds no locally registered class for the config's `model_type`, so
    registering it once turns those nested loads into silent local lookups.
    """
    try:
        AutoConfig.register(config.model_type, type(config))
    except ValueError:
        # Already registered by an earlier load in this process.
        pass


class MageVLHF:
    """Mage-VL — codec-native streaming VLM (Mage-ViT encoder + Qwen3-4B backbone).

    Accepts images as `[0, 255]` numpy arrays / torch tensors (or lists thereof),
    matching the other VLMs in this package, and videos as a path to a container
    file that ffmpeg can demux.

    Video is prompted through the codec backend by default: rather than sampling
    frames uniformly, the codec's per-macroblock bitcost picks the informative
    patches and packs them into canvases. Two engines are supported:

    * ``hevc`` (default) — the ``cv-preinfer`` binary shipped as a console script
      by the ``codec-video-prep`` package. Runs on CPU. Requires the binary on
      ``PATH``, or ``CV_PREINFER_BIN`` pointing at it.
    * ``dcvc-rt`` — the neural codec bundled in the model package under
      ``neural_codec/``. Roughly an order of magnitude slower unless its CUDA
      kernels are compiled, because it decodes every frame up to the last sampled
      one to keep temporal references valid.
    """

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        quantization_config: Optional["BitsAndBytesConfig"] = None,  # noqa: F821
        **kwargs,
    ) -> "MageVLHF":
        """Load Mage-VL from a model package directory.

        Unlike the other package-local-code models here (moondream2, florence-2),
        this one cannot be loaded through `import_class_from_file`: the checkpoint's
        modules import each other relatively (`from .configuration_mage_vl import
        ...`, and a lazy `from .streammind_gate import ...` inside a forward path),
        which only resolves if the package directory is set up as a real Python
        package. So transformers' own dynamic-module loader does the work. It is
        the same code either way, executed from the same local package directory —
        `local_files_only` keeps the loader off the hub.
        """
        # Assert the package-local code is present before handing the directory to
        # transformers, so a truncated package fails here rather than mid-load.
        get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=[
                "configuration_mage_vl.py",
                "modeling_mage_vl.py",
                "processing_mage_vl.py",
                "video_processing_mage_vl.py",
                "codec_video_processing_mage_vl.py",
            ],
        )
        # `AutoModelForCausalLM.from_pretrained` pops `trust_remote_code` before it
        # delegates to `AutoConfig.from_pretrained`, so the nested config load would
        # see `None` and block on an interactive `input()` prompt (15s SIGALRM, then
        # a hard failure). Resolving the config here skips that branch entirely.
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        _register_config_class(config)
        attn_implementation = _get_mage_vl_attn_implementation(device)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            local_files_only=local_files_only,
            device_map=device,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            dtype=cls.default_dtype,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        return cls(
            model=model,
            processor=processor,
            model_package_dir=model_name_or_path,
            device=device,
        )

    def __init__(
        self,
        model,
        processor,
        model_package_dir: str,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._model_package_dir = model_package_dir
        self._device = device
        self._lock = Lock()

    def prompt(
        self,
        images: Optional[ImageInput] = None,
        video: Optional[VideoInput] = None,
        prompt: str = None,
        input_color_format: Optional[ColorFormat] = None,
        max_new_tokens: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_MAGE_VL_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        codec_engine: str = INFERENCE_MODELS_MAGE_VL_DEFAULT_CODEC_ENGINE,
        target_canvas: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_TARGET_CANVAS,
        max_pixels: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_PIXELS,
        **kwargs,
    ) -> List[str]:
        inputs = self.pre_process_generation(
            images=images,
            video=video,
            prompt=prompt,
            input_color_format=input_color_format,
            codec_engine=codec_engine,
            target_canvas=target_canvas,
            max_pixels=max_pixels,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def pre_process_generation(
        self,
        images: Optional[ImageInput] = None,
        video: Optional[VideoInput] = None,
        prompt: str = None,
        input_color_format: Optional[ColorFormat] = None,
        codec_engine: str = INFERENCE_MODELS_MAGE_VL_DEFAULT_CODEC_ENGINE,
        target_canvas: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_TARGET_CANVAS,
        max_pixels: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_PIXELS,
        **kwargs,
    ) -> dict:
        if (images is None) == (video is None):
            raise ModelInputError(
                message="Mage-VL requires exactly one of `images` or `video` to be provided.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        prompt = prompt or "Describe what you see."
        if images is not None:
            return self._pre_process_images(
                images=images, prompt=prompt, input_color_format=input_color_format
            )
        return self._pre_process_video(
            video=video,
            prompt=prompt,
            codec_engine=codec_engine,
            target_canvas=target_canvas,
            max_pixels=max_pixels,
        )

    def _pre_process_images(
        self,
        images: ImageInput,
        prompt: str,
        input_color_format: Optional[ColorFormat],
    ) -> dict:
        images = _to_rgb(images=images, input_color_format=input_color_format)
        text = self._chat_text(media_type="image", prompt=prompt)
        model_inputs = self._processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True,
        )
        return self._to_device(model_inputs)

    def _pre_process_video(
        self,
        video: VideoInput,
        prompt: str,
        codec_engine: str,
        target_canvas: int,
        max_pixels: int,
    ) -> dict:
        if codec_engine not in CODEC_ENGINES:
            raise ModelInputError(
                message=f"Unknown Mage-VL codec engine '{codec_engine}'. "
                f"Supported engines: {sorted(CODEC_ENGINES)}.",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        video = os.fspath(video)
        if not os.path.isfile(video):
            raise ModelInputError(
                message=f"Video file does not exist: {video}",
                help_url="https://inference-models.roboflow.com/errors/models-input/#modelinputerror",
            )
        codec_config = {
            "engine": codec_engine,
            "target_canvas": target_canvas,
            "patch": CODEC_PATCH_SIZE,
        }
        if codec_engine == "dcvc-rt":
            codec_config["dcvc"] = {
                "pkg_dir": os.path.join(self._model_package_dir, "neural_codec"),
                "device": str(self._device),
            }
        try:
            model_inputs = self._processor(
                text=[self._chat_text(media_type="video", prompt=prompt)],
                videos=[video],
                video_backend="codec",
                max_pixels=max_pixels,
                codec_config=codec_config,
                return_tensors="pt",
                padding=True,
            )
        except (ImportError, FileNotFoundError) as error:
            # The common misconfigurations: `codec-video-prep` not installed
            # (ImportError), or the `cv-preinfer` / `ffmpeg` / `ffprobe` binaries
            # not on PATH (FileNotFoundError from the subprocess layer).
            raise MissingDependencyError(
                message=f"Codec pre-processing of the video failed for engine "
                f"'{codec_engine}': {error}. {MAGE_VL_VIDEO_DEPS_HINT}",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#missingdependencyerror",
            ) from error
        except Exception as error:
            raise ModelRuntimeError(
                message=f"Codec pre-processing of the video failed for engine "
                f"'{codec_engine}': {error}",
                help_url="https://inference-models.roboflow.com/errors/models-runtime/#modelruntimeerror",
            ) from error
        return self._to_device(model_inputs)

    def _chat_text(self, media_type: str, prompt: str) -> str:
        conversation = [
            {
                "role": "user",
                "content": [{"type": media_type}, {"type": "text", "text": prompt}],
            }
        ]
        return self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )

    def _to_device(self, model_inputs) -> dict:
        model_inputs = {
            key: (value.to(self._device) if hasattr(value, "to") else value)
            for key, value in model_inputs.items()
        }
        if "pixel_values" in model_inputs:
            model_inputs["pixel_values"] = model_inputs["pixel_values"].to(
                self._model.dtype
            )
        return model_inputs

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = INFERENCE_MODELS_MAGE_VL_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_MAGE_VL_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        input_len = inputs["input_ids"].shape[-1]
        with self._lock, torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=self._processor.tokenizer.pad_token_id,
                eos_token_id=self._processor.tokenizer.eos_token_id,
            )
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
        decoded = self._processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return [text.strip() for text in decoded]


def _to_rgb(
    images: ImageInput,
    input_color_format: Optional[ColorFormat],
) -> ImageInput:
    """Flip BGR inputs to the RGB the processor expects.

    An explicit "bgr" flips every input, tensors included, like the sibling VLMs.
    When no format is declared, numpy inputs are assumed BGR (the cv2 default)
    and flipped; tensors are passed through unchanged.
    """
    if input_color_format == "rgb":
        return images
    flip_tensors = input_color_format == "bgr"
    if isinstance(images, Sequence) and not isinstance(images, (str, bytes)):
        return [_flip_to_rgb(image, flip_tensors=flip_tensors) for image in images]
    return _flip_to_rgb(images, flip_tensors=flip_tensors)


def _flip_to_rgb(image, flip_tensors: bool):
    if isinstance(image, np.ndarray):
        return image[..., ::-1].copy()
    if flip_tensors and isinstance(image, torch.Tensor) and image.ndim == 3:
        if image.shape[0] == 3:  # CHW
            return image[[2, 1, 0], :, :]
        if image.shape[-1] == 3:  # HWC
            return image[..., [2, 1, 0]]
    return image
