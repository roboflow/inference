from typing import List

import torch

from inference.core.env import DEVICE
from inference.core.models.roboflow import RoboflowCoreModel

if DEVICE is None:
    if torch.cuda.is_available():
        DEVICE = "cuda:0"
    elif torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"


class EasyOCR(RoboflowCoreModel):
    """Roboflow EasyOCR model implementation.

    This class is responsible for handling the EasyOCR model, including
    loading the model, preprocessing the input, and performing inference.
    """

    def __init__(
        self,
        model_id: str = "easy_ocr/english_g2",
        device: str = DEVICE,
        *args,
        **kwargs,
    ):
        """Initializes EasyOCR with the given arguments and keyword arguments."""

        super().__init__(model_id=model_id.lower(), *args, **kwargs)
        self.device = device

        self.log("Creating EasyOCR model")

    def get_infer_bucket_file_list(self) -> List[str]:
        return ["weights.pt"]
