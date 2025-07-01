import os
from functools import partial
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import json

from inference_exp.configuration import DEFAULT_DEVICE
import inference_exp.models.perception_encoder.vision_encoder.pe as pe
import inference_exp.models.perception_encoder.vision_encoder.transforms as transforms
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.common.model_packages import get_model_package_contents


# based on original implementation using PIL images found in vision_encoder/transforms.py
# but adjusted to work directly on tensors
def get_tensor_image_transform(
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

    return T.Compose(
        crop
        + [
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True),
        ]
    )


def create_preprocessor(image_size: int) -> Callable:
    preprocessor = get_tensor_image_transform(image_size)

    def _preprocess_image(image: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(image, np.ndarray):
            # HWC -> CHW
            image = torch.from_numpy(image).permute(2, 0, 1)
            # BGR -> RGB
            image = image[[2, 1, 0], :, :]

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                "Unsupported image type, must be np.ndarray or torch.Tensor"
            )

        # The original ToTensor() transform also scaled images from [0, 255] to [0, 1].
        # We need to replicate that behavior.
        if image.dtype == torch.uint8:
            image = image.to(torch.float32) / 255.0

        preprocessed_image = preprocessor(image)
        return preprocessed_image.unsqueeze(0)

    return _preprocess_image


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
        with open(model_config_file) as f:
            config = json.load(f)

        print(
            "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
        )
        print(config["vision_encoder_config"])
        print(model_weights_file)

        model = pe.CLIP.from_config(
            config["vision_encoder_config"],
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
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(images, list):
            imgs = [self.preprocessor(i) for i in images]
            img_in = torch.cat(imgs, dim=0).to(self.device)
        else:
            img_in = self.preprocessor(images).to(self.device)

        if self.device.type == "cpu" or self.device.type == "mps":
            with torch.inference_mode():
                image_features, _, _ = self.model(img_in, None)
                embeddings = image_features.float()
        else:
            with torch.inference_mode(), torch.autocast(self.device.type):
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
            with torch.no_grad():
                _, text_features, _ = self.model(None, tokenized)
        else:
            with torch.inference_mode(), torch.autocast(self.device.type):
                _, text_features, _ = self.model(None, tokenized)

        embeddings = text_features.float()
        return embeddings
