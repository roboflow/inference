from typing import List, Tuple, Union

import numpy as np
import torch

from inference.v1 import Detections
from inference.v1.entities import ImageDimensions
from inference.v1.models.base.object_detection import OpenVocabularyObjectDetectionModel
from inference.v1.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)


class GroundingDinoForObjectDetectionTorch(
    OpenVocabularyObjectDetectionModel[torch.Tensor, ImageDimensions, torch.Tensor]
):
    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "GroundingDinoForObjectDetectionTorch":
        pass

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    def forward(
        self,
        pre_processed_images: PreprocessedInputs,
        classes_or_caption: Union[str, List[str]],
        **kwargs,
    ) -> RawPrediction:
        pass

    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        pass
