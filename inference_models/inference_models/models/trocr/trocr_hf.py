import json
import os
from threading import Lock
from typing import List, Type, Union

import numpy as np
import torch
import transformers
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
)

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.models.base.documents_parsing import TextOnlyOCRModel


class TROcrHF(TextOnlyOCRModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        local_files_only: bool = True,
        **kwargs,
    ) -> "TextOnlyOCRModel":
        model = VisionEncoderDecoderModel.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        ).to(device)
        processor = _load_processor(
            model_name_or_path=model_name_or_path, local_files_only=local_files_only
        )
        return cls(model=model, processor=processor, device=device)

    def __init__(
        self,
        processor: TrOCRProcessor,
        model: VisionEncoderDecoderModel,
        device: torch.device,
    ):
        self._processor = processor
        self._model = model
        self._device = device
        self._lock = Lock()

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        inputs = self._processor(images=images, return_tensors="pt")
        return inputs["pixel_values"].to(self._device)

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        with self._lock, torch.inference_mode():
            return self._model.generate(pre_processed_images)

    def post_process(self, model_results: torch.Tensor, **kwargs) -> List[str]:
        decoded = self._processor.batch_decode(model_results, skip_special_tokens=True)
        return decoded


def _load_processor(model_name_or_path: str, local_files_only: bool) -> TrOCRProcessor:
    # TrOCRProcessor.from_pretrained() cannot be used here: transformers routes
    # vision-encoder-decoder checkpoints to the generic TokenizersBackend
    # (MODELS_WITH_INCORRECT_HUB_TOKENIZER_CLASS), ignoring the class declared in
    # tokenizer_config.json. The generic backend requires tokenizer.json, while
    # Roboflow model packages ship only the sentencepiece serialization.
    image_processor = AutoImageProcessor.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    tokenizer_class = _resolve_tokenizer_class(model_name_or_path=model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        model_name_or_path, local_files_only=local_files_only
    )
    return TrOCRProcessor(image_processor=image_processor, tokenizer=tokenizer)


def _resolve_tokenizer_class(model_name_or_path: str) -> Type:
    tokenizer_config_path = os.path.join(model_name_or_path, "tokenizer_config.json")
    if not os.path.isfile(tokenizer_config_path):
        return AutoTokenizer
    with open(tokenizer_config_path) as f:
        tokenizer_class_name = json.load(f).get("tokenizer_class")
    if not tokenizer_class_name:
        return AutoTokenizer
    tokenizer_class = getattr(transformers, tokenizer_class_name, None)
    if tokenizer_class is None and tokenizer_class_name.endswith("Fast"):
        # packages serialized with transformers 4.x may declare the *Fast variant
        # which transformers 5.x no longer exports
        tokenizer_class = getattr(
            transformers, tokenizer_class_name[: -len("Fast")], None
        )
    if tokenizer_class is None:
        return AutoTokenizer
    return tokenizer_class
