from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch


@dataclass
class ClassificationPrediction:
    class_ids: torch.Tensor  # (bs, )
    confidence: torch.Tensor  # (bs, )
    images_metadata: Optional[List[dict]] = None  # if given, list of size equal to bs


class ClassificationModel(ABC):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *args, **kwargs
    ) -> "ClassificationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> ClassificationPrediction:
        pre_processed_images, pre_processing_meta = self.pre_process(
            images, *args, **kwargs
        )
        model_results = self.forward(pre_processed_images, *args, **kwargs)
        return self.post_process(model_results, *args, **kwargs)

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(
        self, model_results: torch.Tensor, *args, **kwargs
    ) -> ClassificationPrediction:
        pass

    def __call__(
        self, images: torch.Tensor, *args, **kwargs
    ) -> ClassificationPrediction:
        return self.infer(images, *args, **kwargs)


@dataclass
class MultiLabelClassificationPrediction:
    class_ids: torch.Tensor  # (predicted_labels_ids, )
    confidence: torch.Tensor  # (predicted_labels_confidence, )
    image_metadata: Optional[dict] = None


class MultiLabelClassificationModel(ABC):

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, *args, **kwargs
    ) -> "MultiLabelClassificationModel":
        pass

    @property
    @abstractmethod
    def class_names(self) -> List[str]:
        pass

    def infer(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> List[MultiLabelClassificationPrediction]:
        pre_processed_images = self.pre_process(images, *args, **kwargs)
        model_results = self.forward(pre_processed_images, *args, **kwargs)
        return self.post_process(model_results, *args, **kwargs)

    @abstractmethod
    def pre_process(
        self, images: Union[torch.Tensor, List[torch.Tensor]], *args, **kwargs
    ) -> Tuple[torch.Tensor, Any]:
        pass

    @abstractmethod
    def forward(
        self, pre_processed_images: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def post_process(
        self, model_results: torch.Tensor, *args, **kwargs
    ) -> List[MultiLabelClassificationPrediction]:
        pass

    def __call__(
        self, images: torch.Tensor, *args, **kwargs
    ) -> List[MultiLabelClassificationPrediction]:
        return self.infer(images, *args, **kwargs)
