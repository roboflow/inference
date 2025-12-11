import os.path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from inference_exp import Detections, ObjectDetectionModel
from inference_exp.models.auto_loaders.entities import AnyModel
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)
from inference_exp.models.owlv2.owlv2_hf import OWLv2HF


class RoboflowInstantHF(ObjectDetectionModel):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        model_dependencies: Optional[Dict[str, AnyModel]] = None,
        **kwargs,
    ) -> "ObjectDetectionModel":
        model_dependencies = model_dependencies or {}
        if "feature_extractor" in model_dependencies:
            feature_extractor: OWLv2HF = model_dependencies["feature_extractor"]
        else:
            feature_extractor = OWLv2HF.from_pretrained(
                os.path.join(
                    model_name_or_path, "model_dependencies", "feature_extractor"
                )
            )
        return cls(
            feature_extractor=feature_extractor,
        )

    def __init__(self, feature_extractor: OWLv2HF):
        self._feature_extractor = feature_extractor

    @property
    def class_names(self) -> List[str]:
        return ["a", "b", "c"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
        pass

    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    def post_process(
        self,
        model_results: RawPrediction,
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> List[Detections]:
        pass
