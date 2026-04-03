from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, List, Optional, Union

import numpy as np
import torch

from inference_models.models.base.types import PreprocessedInputs, RawPrediction


@dataclass
class ClassificationPrediction:
    class_id: torch.Tensor  # (bs, )
    confidence: torch.Tensor  # (bs, )
    images_metadata: Optional[List[dict]] = None  # if given, list of size equal to bs


class ClassificationModel(ABC, Generic[PreprocessedInputs, RawPrediction]):

    # Single-label classification deliberately opts out of recommendedParameters.
    # Top-1 always wins regardless of confidence, so per-class refinement isn't
    # a meaningful semantic for this task type. (Multi-label classification opts
    # in below — that's where per-class thresholds actually filter the result.)

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "ClassificationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    @property
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size the model supports, or ``None`` if unlimited."""
        return getattr(self, "_max_batch_size", None)

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> ClassificationPrediction:
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
        self, model_results: RawPrediction, **kwargs
    ) -> ClassificationPrediction:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> ClassificationPrediction:
        return self.infer(images, **kwargs)


@dataclass
class MultiLabelClassificationPrediction:
    class_ids: torch.Tensor  # (predicted_labels_ids, )
    confidence: torch.Tensor  # (predicted_labels_confidence, )
    image_metadata: Optional[dict] = None


class MultiLabelClassificationModel(ABC, Generic[PreprocessedInputs, RawPrediction]):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "MultiLabelClassificationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    @property
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size the model supports, or ``None`` if unlimited."""
        return getattr(self, "_max_batch_size", None)

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[MultiLabelClassificationPrediction]:
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
        self, model_results: RawPrediction, **kwargs
    ) -> List[MultiLabelClassificationPrediction]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[MultiLabelClassificationPrediction]:
        return self.infer(images, **kwargs)
