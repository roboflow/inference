from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass(frozen=True)
class SAMImageEmbeddings:
    image_hash: str
    image_size_hw: Tuple[int, int]
    embeddings: torch.Tensor
