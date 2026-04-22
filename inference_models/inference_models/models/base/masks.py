from dataclasses import dataclass
from typing import Literal, Tuple, List

import torch

MaskFormat = Literal["dense", "rle", "compact-rle"]


@dataclass
class RLEMask:
    """
    cocotools RLE format, apart from having single `image_shape`
    for all masks of the same size
    """
    image_shape: Tuple[int, int]  # (h, w)
    rles: List[bytes]


@dataclass
class CompactMask:
    image_shape: Tuple[int, int]  # (h, w)
    rles: List[torch.Tensor]
    crop_shapes: torch.Tensor  # (N,2): (h,w)
    offsets: torch.Tensor  # (N,2): (x1,y1)
