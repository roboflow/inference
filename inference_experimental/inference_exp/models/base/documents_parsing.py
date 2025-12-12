from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, Union

import numpy as np
import torch
from inference_exp.models.base.object_detection import Detections
from inference_exp.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)


class StructuredOCRModel(
    ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "StructuredOCRModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        pre_processed_images, pre_processing_meta = self.pre_process(images, **kwargs)
        model_results = self.forward(pre_processed_images, **kwargs)
        return self.post_process(model_results, pre_processing_meta, **kwargs)

    @abstractmethod
    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[PreprocessedInputs, PreprocessingMetadata]:
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
        pre_processing_meta: PreprocessingMetadata,
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> Tuple[List[str], List[Detections]]:
        return self.infer(images, **kwargs)


class TextOnlyOCRModel(ABC, Generic[PreprocessedInputs, RawPrediction]):
    @classmethod
    @abstractmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs) -> "TextOnlyOCRModel":
        pass

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[str]:
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
        **kwargs,
    ) -> List[str]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[str]:
        return self.infer(images, **kwargs)
