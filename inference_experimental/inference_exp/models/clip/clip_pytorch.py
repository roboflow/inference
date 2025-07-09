import json
from typing import Callable, List, Union

import numpy as np
import torch
import torchvision.transforms as T
import clip
from PIL import Image

from pydantic import BaseModel, ValidationError

from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.common.model_packages import get_model_package_contents


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


def create_preprocessor(image_size: int) -> Callable:
    resize_transform = T.Resize(
        image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True
    )
    center_crop_transform = T.CenterCrop(image_size)
    normalize_transform = T.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    def _preprocess(
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
    ) -> torch.Tensor:
        def _to_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            if isinstance(image, np.ndarray):
                # HWC -> CHW
                image = torch.from_numpy(image).permute(2, 0, 1)
                # BGR -> RGB
                image = image[[2, 1, 0], :, :]
            return image

        if isinstance(images, list):
            # Handle lists of varied-size images by processing them one-by-one
            images_to_stack = []
            for img in images:
                tensor = _to_tensor(img)
                if tensor.dtype == torch.uint8:
                    tensor = tensor.to(torch.float32) / 255.0
                resized = resize_transform(tensor)
                cropped = center_crop_transform(resized)
                images_to_stack.append(cropped)
            tensor_batch = torch.stack(images_to_stack, dim=0)
        else:
            # Handle single image or 4D batch for optimized processing
            tensor_batch = _to_tensor(images)
            if tensor_batch.ndim == 3:
                tensor_batch = tensor_batch.unsqueeze(0)  # Ensure batch dimension
            if tensor_batch.dtype == torch.uint8:
                tensor_batch = tensor_batch.to(torch.float32) / 255.0
            tensor_batch = resize_transform(tensor_batch)
            tensor_batch = center_crop_transform(tensor_batch)

        # Normalize the entire batch
        transformed_batch = normalize_transform(tensor_batch)
        return transformed_batch

    return _preprocess


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
        self.preprocessor = create_preprocessor(model.visual.input_resolution)

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
        **kwargs,
    ) -> torch.Tensor:
        tensor_batch = self.preprocessor(images)
        with torch.no_grad():
            image_features = self.model.encode_image(tensor_batch.to(self.device))
            image_features /= image_features.norm(dim=-1, keepdim=True)

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
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features
