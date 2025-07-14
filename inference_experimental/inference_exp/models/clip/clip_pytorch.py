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
    # This transform pipeline operates on PIL Images, matching the original CLIP implementation
    pil_transform = T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    def _preprocess(
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]
    ) -> torch.Tensor:
        def _to_pil(image: Union[np.ndarray, torch.Tensor]) -> Image.Image:
            if isinstance(image, torch.Tensor):
                # Assuming CHW RGB tensor
                image_numpy = image.permute(1, 2, 0).cpu().numpy()
                if image_numpy.dtype in [np.float32, np.float64]:
                    image_numpy = (image_numpy * 255).astype(np.uint8)
                return Image.fromarray(image_numpy)

            # Assuming HWC BGR numpy array
            if image.ndim == 3 and image.shape[2] == 3:
                image = image[:, :, ::-1]  # BGR to RGB
            return Image.fromarray(image)

        if isinstance(images, list):
            # Handle lists of varied-size images by processing them one-by-one
            images_to_stack = [pil_transform(_to_pil(img)) for img in images]
            tensor_batch = torch.stack(images_to_stack, dim=0)
        elif images.ndim == 4:  # Handle 4D numpy batch
            images_to_stack = [pil_transform(_to_pil(img)) for img in images]
            tensor_batch = torch.stack(images_to_stack, dim=0)
        else:  # Handle single image
            pil_image = _to_pil(images)
            tensor_batch = pil_transform(pil_image).unsqueeze(0)

        return tensor_batch

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
