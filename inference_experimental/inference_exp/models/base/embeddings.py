from abc import ABC, abstractmethod
from typing import List, Literal, Union

import numpy as np
import torch
import torch.nn.functional as F


class TextImageEmbeddingModel(ABC):

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "TextImageEmbeddingModel":
        pass

    def compare_embeddings(
        self,
        x: Union[
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
            str,
            List[str],
        ],
        y: Union[
            torch.Tensor,
            List[torch.Tensor],
            np.ndarray,
            List[np.ndarray],
            str,
            List[str],
        ],
        x_type: Literal["image", "text"] = "image",
        y_type: Literal["image", "text"] = "text",
        **kwargs,
    ) -> torch.Tensor:
        if x_type == "image":
            x_embeddings = self.embed_images(images=x, **kwargs)
        else:
            x_embeddings = self.embed_text(texts=x, **kwargs)
        if y_type == "image":
            y_embeddings = self.embed_images(images=y, **kwargs)
        else:
            y_embeddings = self.embed_text(texts=y, **kwargs)
        x_embeddings_norm = F.normalize(x_embeddings, p=2, dim=1)
        y_embeddings_morm = F.normalize(y_embeddings, p=2, dim=1)
        return x_embeddings_norm @ y_embeddings_morm.T

    @abstractmethod
    def embed_images(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def embed_text(
        self,
        texts: Union[str, List[str]],
        **kwargs,
    ) -> torch.Tensor:
        pass
