from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Optional, Tuple, Union

import numpy as np
import torch

from inference_models.models.base.task_dispatch import ManagedModel, TaskSpec
from inference_models.models.base.types import (
    PreprocessedInputs,
    PreprocessingMetadata,
    RawPrediction,
)


class DepthEstimationModel(
    ManagedModel, ABC, Generic[PreprocessedInputs, PreprocessingMetadata, RawPrediction]
):

    @classmethod
    def get_supported_tasks(cls) -> Dict[str, TaskSpec]:
        return {"infer": TaskSpec(method="infer", default=True, params=["images"])}

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "DepthEstimationModel":
        pass

    @property
    def max_batch_size(self) -> Optional[int]:
        """Maximum batch size the model supports, or ``None`` if unlimited."""
        return getattr(self, "_max_batch_size", None)

    def infer(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[torch.Tensor]:
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
    ) -> List[torch.Tensor]:
        pass

    def __call__(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        **kwargs,
    ) -> List[torch.Tensor]:
        return self.infer(images, **kwargs)
