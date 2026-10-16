from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Union

import numpy as np
import torch

from inference_models.models.base.types import PreprocessedInputs, RawPrediction


@dataclass
class AnomalyDetectionPrediction:
    """Verdict of an anomaly detection model (PatchCore, FoundAD) for a batch.

    There is no class confidence: the model speaks the raw anomaly score, the
    threshold in effect and the decision ``anomaly_score >= anomaly_threshold``.
    """

    anomaly_scores: torch.Tensor  # (bs, ) raw image scores, larger = more anomalous
    anomaly_threshold: float  # threshold in effect: saved with the model or overridden
    is_anomalous: torch.Tensor  # (bs, ) bool, anomaly_scores >= anomaly_threshold
    anomaly_maps: Optional[torch.Tensor] = (
        None  # (bs, image_size, image_size) float32 local evidence, when requested
    )

    def __len__(self) -> int:
        return int(self.anomaly_scores.shape[0])


class AnomalyDetectionModel(ABC, Generic[PreprocessedInputs, RawPrediction]):
    """Anomaly detection models are registered under the classification task
    (``normal`` / ``anomalous``) but predict a score and a threshold, never a
    class confidence."""

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "AnomalyDetectionModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> AnomalyDetectionPrediction:
        pre_processed_images = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, **kwargs)

    @abstractmethod
    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> PreprocessedInputs:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: PreprocessedInputs, **kwargs
    ) -> RawPrediction:
        pass

    @abstractmethod
    def post_process(
        self,
        model_results: RawPrediction,
        include_anomaly_map: bool = False,
        anomaly_threshold: Optional[float] = None,
        **kwargs,
    ) -> AnomalyDetectionPrediction:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> AnomalyDetectionPrediction:
        return self.infer(images, **kwargs)
