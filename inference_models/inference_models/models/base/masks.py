from dataclasses import dataclass
from typing import Tuple


@dataclass
class RLEMask:
    """
    cocotools RLE format, just unwrapped into dataclass
    """
    size: Tuple[int, int]  # (h, w)
    counts: bytes