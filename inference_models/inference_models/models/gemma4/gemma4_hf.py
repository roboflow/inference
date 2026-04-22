import os
from threading import Lock
from typing import Any, Final, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_GEMMA4_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_GEMMA4_DEFAULT_ENABLE_THINKING,
    INFERENCE_MODELS_GEMMA4_DEFAULT_MAX_NEW_TOKENS,
    INFERENCE_MODELS_GEMMA4_DEFAULT_SKIP_SPECIAL_TOKENS,
    INFERENCE_MODELS_GEMMA4_DEFAULT_TEMPERATURE,
    INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_K,
    INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_P,
)
from inference_models.entities import ColorFormat
from inference_models.errors import InvalidModelInitParameterError
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)


# HF Supported budgets for tokens representing images.
# A higher token budget preserves more visual detail at the cost of additional compute,
# while a lower budget enables faster inference for tasks that don't require fine-grained understanding
_GEMMA4_IMAGE_TOKEN_BUDGETS: Final[FrozenSet[int]] = (
    frozenset({70, 140, 280, 560, 1120})
)

# Per-image soft-token counts produced by the HF vision preprocessor when building
# multimodal prompts. Used to expand image placeholders in text; not a tensor argument
# to the transformer (Hugging Face ``transformers`` multimodal ``ProcessorMixin`` stack).
_PROCESSOR_NUM_SOFT_TOKENS_PER_IMAGE_KEY: Final[str] = "num_soft_tokens_per_image"

# Same role as ``PROCESSOR_NUM_SOFT_TOKENS_PER_IMAGE_KEY`` for video segments.
_PROCESSOR_NUM_SOFT_TOKENS_PER_VIDEO_KEY: Final[str] = "num_soft_tokens_per_video"

# Tokenizer output mapping token indices to source character spans (when requested).
# Never a model forward kwarg; strip if present so ``generate(**batch)`` does not fail.
_TOKENIZER_OFFSET_MAPPING_KEY: Final[str] = "offset_mapping"

# All keys above that we defensively remove before ``self._model.generate(**...)``.
_BATCH_KEYS_TO_STRIP_BEFORE_GENERATE: Final[FrozenSet[str]] = (
    frozenset({
        _PROCESSOR_NUM_SOFT_TOKENS_PER_IMAGE_KEY,
        _PROCESSOR_NUM_SOFT_TOKENS_PER_VIDEO_KEY,
        _TOKENIZER_OFFSET_MAPPING_KEY,
    })
)


def _get_gemma4_attn_implementation(device: torch.device) -> str:
    if is_flash_attn_2_available() and device.type == "cuda":
        try:
            import flash_attn  # noqa: F401

            major, _ = torch.cuda.get_device_capability(device=device)
            if major >= 8:
                return "flash_attention_2"
        except ImportError:
            pass
    return "eager"


def _to_pil_rgb(
    image: Union[np.ndarray, torch.Tensor],
    input_color_format: Optional[ColorFormat],
) -> Image.Image:
    if isinstance(image, torch.Tensor):
        arr = image.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.max() <= 1.0 + 1e-6:
            arr = (arr * 255.0).clip(0, 255)
        arr = arr.astype(np.uint8)
    else:
        arr = np.asarray(image)
        if arr.dtype != np.uint8:
            if np.issubdtype(arr.dtype, np.floating) and arr.max() <= 1.0 + 1e-6:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    if input_color_format == "bgr":
        arr = arr[..., ::-1].copy() if arr.shape[-1] == 3 else arr
    elif input_color_format is None and isinstance(image, np.ndarray):
        arr = arr[..., ::-1].copy() if arr.shape[-1] == 3 else arr
    return Image.fromarray(arr).convert("RGB")


class Gemma4HF:
    """Hugging Face Gemma 4 multimodal (vision + text) instruction-tuned models."""

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Optional[BitsAndBytesConfig] = None,
        disable_quantization: bool = False,
        gemma_image_seq_length: Optional[int] = None,
        **kwargs,
    ) -> "Gemma4HF":
        """Load a Gemma 4 checkpoint from a local directory (or cache path).

        Args:
            model_name_or_path: Directory with model weights and processor files.
            device: Torch device used for ``device_map`` and tensor placement.
            trust_remote_code: Passed through to Hugging Face loaders.
            local_files_only: If True, do not hit the network when resolving files.
            quantization_config: Optional BitsAndBytes config; when None on CUDA,
                a default 4-bit config may be applied unless ``disable_quantization``.
            disable_quantization: When True, skip default 4-bit loading on CUDA.
            gemma_image_seq_length: Overrides the processor's visual token budget for
                each image (``processor.image_seq_length``). Gemma 4 uses this budget
                alongside variable aspect ratios: higher values keep more visual detail
                at higher compute cost; lower values speed up inference when fine detail
                is not needed. Allowed values: ``70``, ``140``, ``280``, ``560``,
                ``1120`` (see Hugging Face model cards). Typical guidance: prefer lower
                budgets for classification, captioning, or many-frame / video-style
                workloads; prefer higher budgets for OCR, documents, or small text.

        Returns:
            An initialized :class:`Gemma4HF` instance.
        """
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        inference_config_path = os.path.join(
            model_name_or_path, "inference_config.json"
        )
        inference_config = None
        if os.path.exists(inference_config_path):
            inference_config = parse_inference_config(
                config_path=inference_config_path,
                allowed_resize_modes={
                    ResizeMode.STRETCH_TO,
                    ResizeMode.LETTERBOX,
                    ResizeMode.CENTER_CROP,
                    ResizeMode.LETTERBOX_REFLECT_EDGES,
                    ResizeMode.FIT_LONGER_EDGE,
                },
            )

        if gemma_image_seq_length is not None and (
            gemma_image_seq_length not in _GEMMA4_IMAGE_TOKEN_BUDGETS
        ):
            raise InvalidModelInitParameterError(
                message=(
                    f"While loading Gemma 4, `gemma_image_seq_length` was set to `{gemma_image_seq_length}` "
                    f"which is invalid. Supported visual token budgets: "
                    f"{sorted(_GEMMA4_IMAGE_TOKEN_BUDGETS)}."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#invalidmodelinitparametererror",
            )

        if (
            quantization_config is None
            and device.type == "cuda"
            and not disable_quantization
        ):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            )

        attn_implementation = _get_gemma4_attn_implementation(device)

        load_kw = dict(
            dtype="auto",
            device_map=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
            attn_implementation=attn_implementation,
        )
        if quantization_config is not None and device.type == "cuda":
            load_kw["quantization_config"] = quantization_config

        if os.path.exists(adapter_config_path):
            base_model_path = os.path.join(model_name_or_path, "base")
            base_model = AutoModelForMultimodalLM.from_pretrained(
                base_model_path,
                **load_kw,
            )
            model = PeftModel.from_pretrained(base_model, model_name_or_path)
            if quantization_config is None:
                model = model.merge_and_unload()
                model.to(device)
            processor_path = (
                os.path.join(model_name_or_path, "base")
                if os.path.isdir(os.path.join(model_name_or_path, "base"))
                else model_name_or_path
            )
        else:
            model = AutoModelForMultimodalLM.from_pretrained(
                model_name_or_path,
                **load_kw,
            )
            processor_path = model_name_or_path

        processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if gemma_image_seq_length is not None:
            processor.image_seq_length = gemma_image_seq_length

        model.eval()

        return cls(
            model=model,
            processor=processor,
            inference_config=inference_config,
            device=device,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        processor: Any,
        inference_config: Optional[InferenceConfig],
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._inference_config = inference_config
        self._device = device
        self.default_system_prompt = (
            "You are Gemma 4, a helpful multimodal assistant. Answer clearly and accurately."
        )
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = INFERENCE_MODELS_GEMMA4_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GEMMA4_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = INFERENCE_MODELS_GEMMA4_DEFAULT_SKIP_SPECIAL_TOKENS,
        enable_thinking: bool = INFERENCE_MODELS_GEMMA4_DEFAULT_ENABLE_THINKING,
        **kwargs,
    ) -> List[str]:
        inputs = self.pre_process_generation(
            images=images,
            prompt=prompt,
            input_color_format=input_color_format,
            enable_thinking=enable_thinking,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            **kwargs,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
        )

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        image_size: Optional[Tuple[int, int]] = None,
        enable_thinking: bool = INFERENCE_MODELS_GEMMA4_DEFAULT_ENABLE_THINKING,
        **kwargs,
    ):
        if self._inference_config is None:

            def _collect_list() -> List[Union[np.ndarray, torch.Tensor]]:
                if isinstance(images, torch.Tensor) and images.ndim == 4:
                    return [images[i] for i in range(images.shape[0])]
                if isinstance(images, list):
                    return images
                return [images]

            raw_list = _collect_list()
        else:
            processed = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
                image_size_wh=image_size,
            )[0]
            raw_list = [t.squeeze(0) for t in torch.split(processed, 1, dim=0)]

        pil_images = [_to_pil_rgb(img, input_color_format) for img in raw_list]

        if prompt is None:
            prompt = "Describe what you see in this image."
            system_prompt = self.default_system_prompt
        else:
            split_prompt = prompt.split("<system_prompt>")
            if len(split_prompt) == 1:
                prompt = split_prompt[0] or "Describe what you see in this image."
                system_prompt = self.default_system_prompt
            else:
                prompt = split_prompt[0] or "Describe what you see in this image."
                system_prompt = split_prompt[1] or self.default_system_prompt

        user_content: List[dict] = [
            {"type": "image", "image": pil} for pil in pil_images
        ]
        user_content.append({"type": "text", "text": prompt})

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {"role": "user", "content": user_content},
        ]

        inputs = self._processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        return inputs.to(self._device)

    def generate(
        self,
        inputs,
        max_new_tokens: int = INFERENCE_MODELS_GEMMA4_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_GEMMA4_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        batch = dict(inputs)
        for _meta_key in _BATCH_KEYS_TO_STRIP_BEFORE_GENERATE:
            batch.pop(_meta_key, None)
        input_len = batch["input_ids"].shape[-1]

        tok = self._processor.tokenizer
        pad_id = getattr(tok, "pad_token_id", None) or tok.eos_token_id
        gen_kw = {
            **batch,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": pad_id,
            "eos_token_id": tok.eos_token_id,
        }
        if do_sample:
            gen_kw.setdefault(
                "temperature",
                kwargs.get("temperature", INFERENCE_MODELS_GEMMA4_DEFAULT_TEMPERATURE),
            )
            gen_kw.setdefault(
                "top_p",
                kwargs.get("top_p", INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_P),
            )
            gen_kw.setdefault(
                "top_k",
                kwargs.get("top_k", INFERENCE_MODELS_GEMMA4_DEFAULT_TOP_K),
            )

        with self._lock, torch.inference_mode():
            generation = self._model.generate(**gen_kw)

        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        decoded = self._processor.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return [text.strip() for text in decoded]
