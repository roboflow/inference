from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass(frozen=True)
class SAM2ImageEmbeddings:
    image_hash: str
    image_size_hw: Tuple[int, int]
    embeddings: torch.Tensor
    high_resolution_features: List[torch.Tensor]

    def to(self, device: torch.device) -> "SAM2ImageEmbeddings":
        return SAM2ImageEmbeddings(
            image_hash=self.image_hash,
            image_size_hw=self.image_size_hw,
            embeddings=self.embeddings.to(device=device),
            high_resolution_features=[
                f.to(device=device) for f in self.high_resolution_features
            ],
        )
