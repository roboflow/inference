import json
from threading import Lock
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from pydantic import BaseModel, ValidationError

import inference_models.models.perception_encoder.vision_encoder.pe as pe
import inference_models.models.perception_encoder.vision_encoder.transforms as transforms
from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.common.model_packages import get_model_package_contents


class PerceptionEncoderConfig(BaseModel):
    vision_encoder_config: str


def load_config(config_path: str) -> PerceptionEncoderConfig:
    config_data = {}
    try:
        with open(config_path) as f:
            config_data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        raise CorruptedModelPackageError(
            message=f"Could not load or parse perception encoder model package config file: {config_path}. Details: {e}",
            help_url="https://todo",
        ) from e
    try:
        config = PerceptionEncoderConfig.model_validate(config_data)
        return config
    except ValidationError as e:
        raise CorruptedModelPackageError(
            f"Failed validate perception encoder model package config file: {config_path}. Details: {e}"
        ) from e


# based on original implementation using PIL images found in vision_encoder/transforms.py
# but adjusted to work directly on tensors
def create_image_resize_transform(
    image_size: int,
    center_crop: bool = False,
    interpolation: T.InterpolationMode = T.InterpolationMode.BILINEAR,
):
    if center_crop:
        crop = [
            T.Resize(image_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(image_size),
        ]
    else:
        # "Squash": most versatile
        crop = [
            T.Resize(
                (image_size, image_size), interpolation=interpolation, antialias=True
            )
        ]
    return T.Compose(crop)


def create_image_normalize_transform():
    return T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)


def create_preprocessor(image_size: int) -> Callable:
    resize_transform = create_image_resize_transform(image_size)
    normalize_transform = create_image_normalize_transform()

    def _preprocess(
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
    ) -> torch.Tensor:
        def _to_tensor(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
            is_numpy = isinstance(image, np.ndarray)
            if is_numpy:
                tensor_image = torch.from_numpy(image).permute(2, 0, 1)
            else:
                tensor_image = image

            # For numpy array inputs, we default to BGR -> RGB conversion for compatibility.
            # For tensor inputs, we only convert if BGR is explicitly specified, otherwise RGB is assumed.
            if input_color_format == "bgr" or (is_numpy and input_color_format is None):
                # BGR -> RGB
                tensor_image = tensor_image[[2, 1, 0], :, :]

            return tensor_image

        if isinstance(images, list):
            # Resize each image individually, then stack to a batch
            resized_images = [resize_transform(_to_tensor(img)) for img in images]
            tensor_batch = torch.stack(resized_images, dim=0)
        else:
            # Handle single image or pre-batched tensor
            tensor_batch = resize_transform(_to_tensor(images))

        # Ensure there is a batch dimension for single images
        if tensor_batch.ndim == 3:
            tensor_batch = tensor_batch.unsqueeze(0)

        # Perform dtype conversion and normalization on the whole batch for efficiency
        if tensor_batch.dtype == torch.uint8:
            tensor_batch = tensor_batch.to(torch.float32) / 255.0

        transformed_batch = normalize_transform(tensor_batch)
        return transformed_batch

    return _preprocess


class PerceptionEncoderTorch(TextImageEmbeddingModel):
    def __init__(
        self,
        model: pe.CLIP,
        device: torch.device,
    ):
        self.model = model
        self.device = device
        self.preprocessor = create_preprocessor(model.image_size)
        self.tokenizer = transforms.get_text_tokenizer(model.context_length)
        self._lock = Lock()

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, device: torch.device = DEFAULT_DEVICE, **kwargs
    ) -> "PerceptionEncoderTorch":
        # here model name came from path before, which maybe doesn't match directly with how our registry works
        # instead should this be adopted to read config file that is served as part of model package?
        # model_config = model_name_or_path.split("/")[-1]
        # checkpoint_path = os.path.join(model_name_or_path, "model.pt")

        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["config.json", "model.pt"],
        )

        model_config_file = model_package_content["config.json"]
        model_weights_file = model_package_content["model.pt"]
        config = load_config(model_config_file)

        model = pe.CLIP.from_config(
            config.vision_encoder_config,
            pretrained=True,
            checkpoint_path=model_weights_file,
        )
        model = model.to(device)
        model.eval()

        return cls(
            model=model,
            device=device,
        )

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        img_in = self.preprocessor(images, input_color_format=input_color_format).to(
            self.device
        )

        if self.device.type == "cpu" or self.device.type == "mps":
            with self._lock, torch.inference_mode():
                image_features, _, _ = self.model(img_in, None)
                embeddings = image_features.float()
        else:
            with self._lock, torch.inference_mode(), torch.autocast(self.device.type):
                image_features, _, _ = self.model(img_in, None)
                embeddings = image_features.float()

        return embeddings

    def embed_text(
        self,
        texts: Union[str, List[str]],
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(texts, list):
            texts_to_embed = texts
        else:
            texts_to_embed = [texts]

        # results = []
        # The original implementation had batching here based on CLIP_MAX_BATCH_SIZE, but not entirely sure how to handle that with Tensor output
        # I will leave it out for now, see https://github.com/roboflow/inference/blob/main/inference/models/perception_encoder/perception_encoder.py#L227
        tokenized = self.tokenizer(texts_to_embed).to(self.device)
        if self.device.type == "cpu" or self.device.type == "mps":
            with self._lock, torch.no_grad():
                _, text_features, _ = self.model(None, tokenized)
        else:
            with self._lock, torch.inference_mode(), torch.autocast(self.device.type):
                _, text_features, _ = self.model(None, tokenized)

        embeddings = text_features.float()
        return embeddings
