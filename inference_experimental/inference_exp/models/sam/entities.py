from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SAMImageEmbeddings:
    image_hash: str
    image_size_hw: Tuple[int, int]
    embeddings: torch.Tensor

    def to(self, device: torch.device) -> "SAMImageEmbeddings":
        return SAMImageEmbeddings(
            image_hash=self.image_hash,
            image_size_hw=self.image_size_hw,
            embeddings=self.embeddings.to(device=device),
        )


@dataclass(frozen=True)
class SAMPrediction:
    masks: torch.Tensor
    scores: torch.Tensor
    logits: torch.Tensor
