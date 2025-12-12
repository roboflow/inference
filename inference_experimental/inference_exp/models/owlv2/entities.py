from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp.models.owlv2.reference_dataset import LazyImageWrapper
from pydantic import BaseModel, ConfigDict, Field


class ReferenceBoundingBox(BaseModel):
    x: Union[float, int]
    y: Union[float, int]
    w: Union[float, int]
    h: Union[float, int]
    cls: str
    negative: bool = Field(default=False)
    absolute: bool = Field(default=True)

    def to_tuple(
        self, image_wh: Optional[Tuple[int, int]] = None
    ) -> Tuple[
        Union[int, float], Union[int, float], Union[int, float], Union[int, float]
    ]:
        if image_wh is None or self.absolute is False:
            return self.x, self.y, self.w, self.h
        max_dim = max(image_wh)
        return (
            self.x / max_dim,
            self.y / max_dim,
            self.w / max_dim,
            self.h / max_dim,
        )


class ReferenceExample(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    image: Union[np.ndarray, torch.Tensor, str, bytes]
    boxes: List[ReferenceBoundingBox]


@dataclass(frozen=True)
class LazyReferenceExample:
    image: LazyImageWrapper
    boxes: List[ReferenceBoundingBox]


@dataclass(frozen=True)
class ImageEmbeddings:
    image_hash: str
    objectness: torch.Tensor
    boxes: torch.Tensor
    image_class_embeddings: torch.Tensor
    logit_shift: torch.Tensor
    logit_scale: torch.Tensor
    image_size_wh: Tuple[int, int]

    def to(self, device: torch.device) -> "ImageEmbeddings":
        return ImageEmbeddings(
            image_hash=self.image_hash,
            objectness=self.objectness.to(device=device),
            boxes=self.boxes.to(device=device),
            image_class_embeddings=self.image_class_embeddings.to(device=device),
            logit_shift=self.logit_shift.to(device=device),
            logit_scale=self.logit_scale.to(device=device),
            image_size_wh=self.image_size_wh,
        )


@dataclass(frozen=True)
class ReferenceExamplesClassEmbeddings:
    positive: Optional[torch.Tensor]
    negative: Optional[torch.Tensor]

    def to(self, device: torch.device) -> "ReferenceExamplesClassEmbeddings":
        return ReferenceExamplesClassEmbeddings(
            positive=(
                self.positive.to(device=device) if self.positive is not None else None
            ),
            negative=(
                self.negative.to(device=device) if self.negative is not None else None
            ),
        )


@dataclass(frozen=True)
class ReferenceExamplesEmbeddings:
    class_embeddings: Dict[str, ReferenceExamplesClassEmbeddings]
    image_embeddings: Optional[Dict[str, ImageEmbeddings]]
