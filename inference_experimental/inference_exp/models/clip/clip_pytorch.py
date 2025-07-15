import json
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import clip
from pydantic import BaseModel, ValidationError

from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.clip.preprocessing import create_clip_preprocessor


class ClipConfig(BaseModel):
    model_name: str


def load_config(config_path: str) -> ClipConfig:
    config_data = {}
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        raise CorruptedModelPackageError(
            message=f"Could not load or parse clip model package config file: {config_path}. Details: {e}",
            help_url="https://todo",
        ) from e
    try:
        config = ClipConfig.model_validate(config_data)
        return config
    except ValidationError as e:
        raise CorruptedModelPackageError(
            f"Failed validate clip model package config file: {config_path}. Details: {e}"
        ) from e


class ClipTorch(TextImageEmbeddingModel):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable,
        tokenizer: Callable,
        device: torch.device,
    ):
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.device = device
        self.preprocessor = create_clip_preprocessor(
            image_size=model.visual.input_resolution, device=device
        )

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, device: torch.device = DEFAULT_DEVICE, **kwargs
    ) -> "ClipTorch":
        model, _ = clip.load(model_name_or_path, device=device)
        tokenizer = clip.tokenize

        return cls(
            model=model,
            preprocess=None,  # Preprocess is created in __init__
            tokenizer=tokenizer,
            device=device,
        )

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        tensor_batch = self.preprocessor(images, input_color_format=input_color_format)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor_batch.to(self.device))

        return image_features

    def embed_text(
        self,
        texts: Union[str, List[str]],
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]

        text_tokens = self.tokenizer(texts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)

        return text_features
