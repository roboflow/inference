from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass(frozen=True)
class SAM3ImageEmbeddings:
    image_hash: str
    image_size_hw: Tuple[int, int]
    embeddings: Dict[str, Any]

    def to(self, device: torch.device) -> "SAM3ImageEmbeddings":
        def _move_to_device(obj: Any) -> Any:
            if isinstance(obj, torch.Tensor):
                return obj.to(device=device)
            elif isinstance(obj, dict):
                return {k: _move_to_device(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_move_to_device(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(_move_to_device(item) for item in obj)
            return obj

        return SAM3ImageEmbeddings(
            image_hash=self.image_hash,
            image_size_hw=self.image_size_hw,
            embeddings=_move_to_device(self.embeddings),
        )


@dataclass(frozen=True)
class SAM3Prediction:
    masks: torch.Tensor
    scores: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True)
class SAM3MaskCacheEntry:
    prompt_hash: str
    serialized_prompt: List[dict]
    mask: torch.Tensor

    def to(self, device: torch.device) -> "SAM3MaskCacheEntry":
        return SAM3MaskCacheEntry(
            prompt_hash=self.prompt_hash,
            serialized_prompt=self.serialized_prompt,
            mask=self.mask.to(device=device),
        )
