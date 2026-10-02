from dataclasses import dataclass
from typing import List, Optional, Tuple, TypeVar

PreprocessedInputs = TypeVar("PreprocessedInputs")
PreprocessingMetadata = TypeVar("PreprocessingMetadata")
RawPrediction = TypeVar("RawPrediction")


@dataclass
class InstancesRLEMasks:
    image_size: Tuple[int, int]  # (h, w) of the image the masks describe
    masks: List[bytes]
    # (h, w) of the grid the masks are encoded on. Defaults to image_size,
    # which is the only value it took before the grid became adjustable.
    mask_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.mask_size is None:
            self.mask_size = self.image_size

    @classmethod
    def from_coco_rle_masks(
        cls,
        image_size: Tuple[int, int],
        masks: List[dict],
        mask_size: Optional[Tuple[int, int]] = None,
    ) -> "InstancesRLEMasks":
        masks = [m["counts"] for m in masks]
        return cls(image_size=image_size, masks=masks, mask_size=mask_size)

    def to_coco_rle_masks(self) -> List[dict]:
        return [{"size": list(self.image_size), "counts": m} for m in self.masks]
