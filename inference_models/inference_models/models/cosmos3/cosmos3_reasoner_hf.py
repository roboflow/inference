"""
This is inference-models wrapper for the reasoner tower of NVIDIA Cosmos 3 Edge,
originally published in https://huggingface.co/nvidia/Cosmos3-Edge
"""

import json
import os
import re
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import transformers
from packaging.version import Version
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.utils import is_flash_attn_2_available

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
    RUNNING_ON_JETSON,
)
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.logger import LOGGER
from inference_models.models.common.roboflow.model_packages import (
    InferenceConfig,
    ResizeMode,
    parse_inference_config,
)
from inference_models.models.common.roboflow.pre_processing import (
    pre_process_network_input,
)

DEFAULT_PROMPT = "Describe what's in this image."
BASE_SYSTEM_PROMPT = (
    "You are Cosmos, a helpful assistant that understands physical scenes "
    "and answers questions about images and videos."
)
# What roboflow-train puts in the system turn of every training conversation
# (src/huggingface/cosmos3edge/image.py SYSTEM_MESSAGE). A fine-tune only
# produces what it was trained to when prompted the same way: with the base
# prompt it answers like the base model. Keep the two in sync.
FINE_TUNE_SYSTEM_PROMPT = (
    "You are Cosmos 3 Edge, a physical AI reasoning model. "
    "Look at the image carefully and answer with only what is asked."
)
# The `cosmos3_edge` model type is what AutoModelForImageTextToText resolves the
# checkpoint to; older transformers fail on it with an unhelpful config error.
MIN_TRANSFORMERS_VERSION = "5.15.0"
SYSTEM_PROMPT_SENTINEL = "<system_prompt>"
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", flags=re.DOTALL)
THINK_EXTRACT_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)


def _get_cosmos3_attn_implementation(device: torch.device) -> str:
    if (
        is_flash_attn_2_available()
        and device
        and "cuda" in str(device)
        and not RUNNING_ON_JETSON
    ):
        try:
            import flash_attn  # noqa: F401

            if _is_ampere_plus(device=device):
                return "flash_attention_2"
            return "eager"
        except ImportError:
            pass
    return "eager"


def _is_ampere_plus(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = torch.cuda.get_device_capability(device=device)
    return major >= 8


def _require_cosmos3_transformers() -> None:
    if Version(transformers.__version__) < Version(MIN_TRANSFORMERS_VERSION):
        raise RuntimeError(
            f"Cosmos 3 Edge needs transformers>={MIN_TRANSFORMERS_VERSION} (the "
            f"cosmos3_edge model type); found {transformers.__version__}."
        )


def _load_inference_config(package_dir: str) -> Optional[InferenceConfig]:
    """The package's inference_config.json when it has one that parses.

    The Cosmos processor sizes images itself, so the config only carries the
    version's photometric steps (auto-orient, grayscale, contrast, static crop).
    roboflow-train currently writes it with training_input_size: null for
    versions without a resize, which the shared schema rejects; that config
    could not drive preprocessing anyway, so it is skipped with a warning
    rather than failing the load.
    """
    config_path = os.path.join(package_dir, "inference_config.json")
    if not os.path.exists(config_path):
        return None
    try:
        return parse_inference_config(
            config_path=config_path,
            allowed_resize_modes={
                ResizeMode.STRETCH_TO,
                ResizeMode.LETTERBOX,
                ResizeMode.CENTER_CROP,
                ResizeMode.LETTERBOX_REFLECT_EDGES,
                ResizeMode.FIT_LONGER_EDGE,
            },
        )
    except CorruptedModelPackageError as error:
        LOGGER.warning(
            f"Ignoring {config_path}: {error.__cause__ or error}. Images reach the "
            "Cosmos 3 Edge processor as-is."
        )
        return None


def _adapter_trains_new_tokens(adapter_config_path: str) -> bool:
    with open(adapter_config_path) as f:
        adapter_config = json.load(f)
    return bool(adapter_config.get("trainable_token_indices"))


def _resolve_default_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


class Cosmos3EdgeReasoner:
    """NVIDIA Cosmos 3 Edge reasoner tower (image/video + text -> text).

    Only the autoregressive reasoner is exposed here; the diffusion generator
    (image-to-video, dynamics, policy) has a separate implementation and
    registry entry.
    """

    default_dtype = torch.bfloat16

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        trust_remote_code: bool = False,
        local_files_only: bool = True,
        quantization_config: Any = None,
        **kwargs,
    ) -> "Cosmos3EdgeReasoner":
        _require_cosmos3_transformers()
        dtype = _resolve_default_dtype(device)
        attn_implementation = _get_cosmos3_attn_implementation(device)
        inference_config = _load_inference_config(model_name_or_path)
        adapter_config_path = os.path.join(model_name_or_path, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            # A Roboflow fine-tune: the LoRA adapter sits at the package root and the
            # base checkpoint (weights, tokenizer, chat template, processor configs)
            # under base/, the same layout as the other fine-tuned VLMs. A video
            # fine-tune adds one class token per class to the vocabulary, rows the
            # base has no room for until it is resized; refuse it readably instead
            # of failing on a shape mismatch inside PEFT.
            if _adapter_trains_new_tokens(adapter_config_path):
                raise NotImplementedError(
                    "This Cosmos 3 Edge fine-tune adds class tokens to the vocabulary "
                    "(a video fine-tune); only image fine-tunes can be served for now."
                )
            base_model_path = os.path.join(model_name_or_path, "base")
            model = AutoModelForImageTextToText.from_pretrained(
                base_model_path,
                device_map=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            )
            model = PeftModel.from_pretrained(model, model_name_or_path)
            if quantization_config is None:
                model = model.merge_and_unload()
            model = model.eval()
            processor = AutoProcessor.from_pretrained(
                base_model_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
            # Roboflow fine-tunes are trained with the think block left empty and
            # the trainer's system prompt, and only answer as trained when served
            # the same way: thinking would eat the token budget before the answer,
            # and the base prompt makes them answer like the base model.
            return cls(
                model=model,
                processor=processor,
                device=device,
                fine_tuned=True,
                inference_config=inference_config,
            )
        else:
            model = AutoModelForImageTextToText.from_pretrained(
                model_name_or_path,
                device_map=device,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
            ).eval()
            processor = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                local_files_only=local_files_only,
            )
        return cls(
            model=model,
            processor=processor,
            device=device,
            inference_config=inference_config,
        )

    def __init__(
        self,
        model,
        processor,
        device: torch.device,
        fine_tuned: bool = False,
        inference_config: Optional[InferenceConfig] = None,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self._fine_tuned = fine_tuned
        self._inference_config = inference_config
        self._enable_thinking = not fine_tuned
        self._torch_dtype = next(model.parameters()).dtype
        self.default_system_prompt = (
            FINE_TUNE_SYSTEM_PROMPT if fine_tuned else BASE_SYSTEM_PROMPT
        )
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        inputs = self.pre_process_generation(
            images=images, prompt=prompt, input_color_format=input_color_format
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            return_thinking=return_thinking,
        )

    def prompt_video(
        self,
        frames: List[np.ndarray],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[str, Dict[str, str]]:
        inputs = self.pre_process_generation(
            images=frames,
            prompt=prompt,
            input_color_format=input_color_format,
            as_video=True,
        )
        generated_ids = self.generate(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        return self.post_process_generation(
            generated_ids=generated_ids,
            skip_special_tokens=skip_special_tokens,
            return_thinking=return_thinking,
        )[0]

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        as_video: bool = False,
        **kwargs,
    ) -> dict:
        if self._inference_config is not None and not as_video:
            # The version's preprocessing (photometric steps, any resize it baked)
            # applied the way the other fine-tuned VLMs do; the result is RGB.
            images = pre_process_network_input(
                images=images,
                image_pre_processing=self._inference_config.image_pre_processing,
                network_input=self._inference_config.network_input,
                target_device=self._device,
                input_color_format=input_color_format,
            )[0]
            images = [frame[0] for frame in torch.split(images, 1, dim=0)]
        elif isinstance(images, np.ndarray):
            if input_color_format != "rgb":
                images = images[:, :, ::-1]
            images = images.copy()
        elif as_video and input_color_format != "rgb":
            images = [
                frame[:, :, ::-1].copy() if isinstance(frame, np.ndarray) else frame
                for frame in images
            ]
        prompt, system_prompt = self._parse_prompt(prompt=prompt)
        visual_content = (
            {"type": "video", "video": images}
            if as_video
            else {"type": "image", "image": images}
        )
        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    visual_content,
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        template_kwargs = {} if self._enable_thinking else {"enable_thinking": False}
        text_input = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True, **template_kwargs
        )
        processor_kwargs = {"videos": [images]} if as_video else {"images": images}
        model_inputs = self._processor(
            text=text_input,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        )
        return {
            k: (
                v.to(self._device, dtype=self._torch_dtype)
                if v.is_floating_point()
                else v.to(self._device)
            )
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

    def generate(
        self,
        inputs: dict,
        max_new_tokens: Optional[int] = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        **kwargs,
    ) -> torch.Tensor:
        if max_new_tokens is None:
            max_new_tokens = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS
        input_len = inputs["input_ids"].shape[-1]
        tokenizer = self._processor.tokenizer
        pad_token_id = (
            getattr(tokenizer, "pad_token_id", None) or tokenizer.eos_token_id
        )
        with self._lock, torch.inference_mode():
            generation = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        return_thinking: bool = False,
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        result = []
        for text in decoded:
            text = text.replace("assistant\n", "")
            if not self._enable_thinking:
                # Served without a reasoning block (a fine-tune): everything the
                # model wrote is the answer, and there is no thinking to return.
                answer = THINK_BLOCK_PATTERN.sub("", text).strip()
                result.append(
                    {"thinking": "", "answer": answer} if return_thinking else answer
                )
                continue
            # The chat template opens the reasoning block inside the prompt, so
            # decoded output carries a bare closing </think>. Restore the
            # opening tag so thinking and answer parse apart (qwen3_5 pattern).
            if "</think>" in text and "<think>" not in text:
                text = "<think>" + text
            if return_thinking:
                think_match = THINK_EXTRACT_PATTERN.search(text)
                if think_match:
                    thinking = think_match.group(1).strip()
                    answer = THINK_BLOCK_PATTERN.sub("", text).strip()
                else:
                    thinking = text.replace("<think>", "").strip()
                    answer = ""
                result.append({"thinking": thinking, "answer": answer})
            else:
                result.append(THINK_BLOCK_PATTERN.sub("", text).strip())
        return result

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Union[List[str], List[Dict[str, str]]]:
        return self.prompt(images, **kwargs)

    def _parse_prompt(self, prompt: Optional[str]) -> Tuple[str, str]:
        if prompt is None:
            return DEFAULT_PROMPT, self.default_system_prompt
        split_prompt = prompt.split(SYSTEM_PROMPT_SENTINEL)
        parsed_prompt = split_prompt[0] or DEFAULT_PROMPT
        if len(split_prompt) == 1:
            return parsed_prompt, self.default_system_prompt
        return parsed_prompt, split_prompt[1] or self.default_system_prompt
