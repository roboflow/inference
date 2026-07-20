from threading import Lock
from typing import List, Optional, Union

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference_models.configuration import (
    DEFAULT_DEVICE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
    INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
)
from inference_models.entities import ColorFormat

DEFAULT_PROMPT = "Describe what's in this image."
SYSTEM_PROMPT_SENTINEL = "<system_prompt>"


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
        **kwargs,
    ) -> "Cosmos3EdgeReasoner":
        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            device_map=device,
            dtype=cls.default_dtype,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        ).eval()
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self,
        model: AutoModelForImageTextToText,
        processor: AutoProcessor,
        device: torch.device,
    ):
        self._model = model
        self._processor = processor
        self._device = device
        self.default_system_prompt = (
            "You are Cosmos, a helpful assistant that understands physical scenes "
            "and answers questions about images and videos."
        )
        self._lock = Lock()

    def prompt(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> List[str]:
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
        )

    def prompt_video(
        self,
        frames: List[np.ndarray],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        max_new_tokens: int = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
        skip_special_tokens: bool = True,
        **kwargs,
    ) -> str:
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
        )[0]

    def pre_process_generation(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        prompt: str = None,
        input_color_format: ColorFormat = None,
        as_video: bool = False,
        **kwargs,
    ) -> dict:
        if isinstance(images, np.ndarray):
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
        text_input = self._processor.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        processor_kwargs = {"videos": [images]} if as_video else {"images": images}
        model_inputs = self._processor(
            text=text_input,
            return_tensors="pt",
            padding=True,
            **processor_kwargs,
        )
        return {
            k: v.to(self._device)
            for k, v in model_inputs.items()
            if isinstance(v, torch.Tensor)
        }

    def generate(
        self,
        inputs: dict,
        max_new_tokens: int = INFERENCE_MODELS_COSMOS3_DEFAULT_MAX_NEW_TOKENS,
        do_sample: bool = INFERENCE_MODELS_COSMOS3_DEFAULT_DO_SAMPLE,
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
                bos_token_id=self._processor.tokenizer.bos_token_id,
            )
        return generation[:, input_len:]

    def post_process_generation(
        self,
        generated_ids: torch.Tensor,
        skip_special_tokens: bool = False,
        **kwargs,
    ) -> List[str]:
        decoded = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
        )
        return [text.replace("assistant\n", "").strip() for text in decoded]

    def _parse_prompt(self, prompt: Optional[str]) -> tuple:
        if prompt is None:
            return DEFAULT_PROMPT, self.default_system_prompt
        split_prompt = prompt.split(SYSTEM_PROMPT_SENTINEL)
        parsed_prompt = split_prompt[0] or DEFAULT_PROMPT
        if len(split_prompt) == 1:
            return parsed_prompt, self.default_system_prompt
        return parsed_prompt, split_prompt[1] or self.default_system_prompt
