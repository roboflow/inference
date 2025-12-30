from typing import Callable, List, Optional, Union

import clip
import numpy as np
import torch
from clip.model import CLIP, build_model

from inference_models.configuration import DEFAULT_DEVICE
from inference_models.entities import ColorFormat
from inference_models.errors import CorruptedModelPackageError
from inference_models.models.base.embeddings import TextImageEmbeddingModel
from inference_models.models.clip.preprocessing import create_clip_preprocessor
from inference_models.models.common.model_packages import get_model_package_contents


class ClipTorch(TextImageEmbeddingModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: torch.device = DEFAULT_DEVICE,
        max_batch_size: int = 32,
        **kwargs,
    ) -> "ClipTorch":
        model_package_content = get_model_package_contents(
            model_package_dir=model_name_or_path,
            elements=["model.pt"],
        )
        model_weights_file = model_package_content["model.pt"]
        model = build_clip_model(model_weights_file=model_weights_file, device=device)
        model.eval()
        return cls(
            model=model,
            tokenizer=clip.tokenize,
            device=device,
            max_batch_size=max_batch_size,
        )

    def __init__(
        self,
        model: CLIP,
        tokenizer: Callable,
        device: torch.device,
        max_batch_size: int,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._preprocessor = create_clip_preprocessor(
            image_size=model.visual.input_resolution
        )
        self._max_batch_size = max_batch_size

    @torch.no_grad()
    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        tensor_batch = self._preprocessor(images, input_color_format, self._device)
        if tensor_batch.shape[0] <= self._max_batch_size:
            return self._model.encode_image(tensor_batch.to(self._device))
        results = []
        for i in range(0, tensor_batch.shape[0], self._max_batch_size):
            batch_input = tensor_batch[i : i + self._max_batch_size].contiguous()
            batch_results = self._model.encode_image(batch_input.to(self._device))
            results.append(batch_results)
        return torch.cat(results, dim=0)

    @torch.no_grad()
    def embed_text(
        self,
        texts: Union[str, List[str]],
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        text_tokens = self._tokenizer(texts).to(self._device)
        if text_tokens.shape[0] <= self._max_batch_size:
            return self._model.encode_text(text_tokens)
        results = []
        for i in range(0, text_tokens.shape[0], self._max_batch_size):
            batch_input = text_tokens[i : i + self._max_batch_size].contiguous()
            batch_results = self._model.encode_text(batch_input)
            results.append(batch_results)
        return torch.cat(results, dim=0)


def build_clip_model(model_weights_file: str, device: torch.device) -> CLIP:
    try:
        # The model file is a JIT archive, so we load it as such
        # and then build a new model from its state dict.
        jit_model = torch.jit.load(model_weights_file, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
        model = build_model(state_dict).to(device)
        if device.type == "cpu":
            model.float()
        return model
    except Exception as e:
        raise CorruptedModelPackageError(
            f"Could not load TorchScript model from {model_weights_file}. Details: {e}"
        ) from e
