from typing import List, Optional, Union

import numpy as np
import torch

from inference_models import ClassificationModel, ClassificationPrediction, ColorFormat


class MyClassificationModel(ClassificationModel[torch.Tensor, torch.Tensor]):

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        **kwargs,
    ) -> "MyClassificationModel":
        return cls()

    @property
    def class_names(self) -> List[str]:
        return ["a", "b"]

    def pre_process(
        self,
        images: Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]],
        input_color_format: Optional[ColorFormat] = None,
        **kwargs,
    ) -> torch.Tensor:
        return torch.empty((10,))

    def forward(self, pre_processed_images: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.empty((10,))

    def post_process(
        self,
        model_results: torch.Tensor,
        **kwargs,
    ) -> ClassificationPrediction:
        return ClassificationPrediction(
            class_id=torch.empty((10,)),
            confidence=torch.empty((10,)),
        )
