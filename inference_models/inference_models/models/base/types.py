from dataclasses import dataclass
from typing import List, Tuple, TypeVar

PreprocessedInputs = TypeVar("PreprocessedInputs")
PreprocessingMetadata = TypeVar("PreprocessingMetadata")
RawPrediction = TypeVar("RawPrediction")


@dataclass
class InstancesRLEMasks:
    image_size: Tuple[int, int]  # (h, w)
    masks: List[bytes]

    @classmethod
    def from_coco_rle_masks(
        cls,
        image_size: Tuple[int, int],
        masks: List[dict],
    ) -> "InstancesRLEMasks":
        masks = [m["counts"] for m in masks]
        return cls(image_size=image_size, masks=masks)

    def to_coco_rle_masks(self) -> List[dict]:
        return [{"size": list(self.image_size), "counts": m} for m in self.masks]
