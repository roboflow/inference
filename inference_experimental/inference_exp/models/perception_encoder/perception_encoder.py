import os
from typing import List, Union

import numpy as np
import torch
from PIL import Image

from inference_exp.configuration import DEFAULT_DEVICE
import inference_exp.models.perception_encoder.vision_encoder.pe as pe
import inference_exp.models.perception_encoder.vision_encoder.transforms as transforms
from inference_exp.models.base.embeddings import TextImageEmbeddingModel



class PerceptionEncoder(TextImageEmbeddingModel):
    def __init__(
        self,
        model: pe.CLIP,
        preprocessor,
        tokenizer,
        device: str,
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, device: torch.device = DEFAULT_DEVICE, **kwargs
    ) -> "PerceptionEncoder":
        #here model name came from path before, which maybe doesn't match directly with how our registry works
        # instead should this be adopted to read config file that is served as part of model package?
        model_config = model_name_or_path.split("/")[-1]
        #cache_dir = get_model_cache_dir(model_name_or_path)
        checkpoint_path = os.path.join(model_name_or_path, "model.pt")
        model = pe.CLIP.from_config(
            model_config, pretrained=True, checkpoint_path=checkpoint_path
        )
        model = model.to(device)
        model.eval()

        preprocessor = transforms.get_image_transform(model.image_size)
        tokenizer = transforms.get_text_tokenizer(model.context_length)
        return cls(
            model=model,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            device=device,
        )

    def _preproc_image(self, image: Union[np.ndarray, torch.Tensor, Image.Image]) -> torch.Tensor:
        """Preprocesses an inference request image."""
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            pil_image = Image.fromarray(image.cpu().numpy())
        else:
            pil_image = image
        preprocessed_image = self.preprocessor(pil_image)
        return preprocessed_image.unsqueeze(0)

    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray], List[Image.Image]],
        **kwargs,
    ) -> torch.Tensor:
        if isinstance(images, list):
            imgs = [self._preproc_image(i) for i in images]
            img_in = torch.cat(imgs, dim=0).to(self.device)
        else:
            img_in = self._preproc_image(images).to(self.device)

        if self.device == "cpu" or self.device == "mps":
            with torch.inference_mode():
                image_features, _, _ = self.model(img_in, None)
                embeddings = image_features.float()
        else:
            with torch.inference_mode(), torch.autocast(self.device):
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

        #results = []
        # The original implementation had batching here based on CLIP_MAX_BATCH_SIZE, but not entirely sure how to handle that with Tensor output
        # I will leave it out for now, see https://github.com/roboflow/inference/blob/main/inference/models/perception_encoder/perception_encoder.py#L227 
        tokenized = self.tokenizer(texts_to_embed).to(self.device)
        if self.device == "cpu" or self.device == "mps":
            with torch.no_grad():
                _, text_features, _ = self.model(None, tokenized)
        else:
            with torch.inference_mode(), torch.autocast(self.device):
                _, text_features, _ = self.model(None, tokenized)

        embeddings = text_features.float()
        return embeddings 