import os
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import clip
from clip.model import build_model

from inference_exp.configuration import DEFAULT_DEVICE
from inference_exp.entities import ColorFormat
from inference_exp.errors import CorruptedModelPackageError
from inference_exp.models.base.embeddings import TextImageEmbeddingModel
from inference_exp.models.clip.preprocessing import create_clip_preprocessor
from inference_exp.models.common.model_packages import get_model_package_contents


class ClipTorch(TextImageEmbeddingModel):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, device: torch.device = DEFAULT_DEVICE, **kwargs
    ) -> "ClipTorch":

        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["model.pt"],
        )
        model_weights_file = model_package_content["model.pt"]

        try:
            # The model file is a JIT archive, so we load it as such
            # and then build a new model from its state dict.
            jit_model = torch.jit.load(model_weights_file, map_location="cpu").eval()
            state_dict = jit_model.state_dict()

            model = build_model(state_dict).to(device)

            if device.type == "cpu":
                model.float()

            model.eval()
        except Exception as e:
            raise CorruptedModelPackageError(
                f"Could not load TorchScript model from {model_weights_file}. Details: {e}"
            ) from e

        return cls(
            model=model,
            tokenizer=clip.tokenize,
            device=device,
        )

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Callable,
        device: torch.device,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.preprocessor = create_clip_preprocessor(
            image_size=model.visual.input_resolution, device=device
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
