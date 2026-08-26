from typing import List, Optional, Union

import numpy as np
import torch

from inference_models.models.base.video_classification import (
    VideoClassificationModel,
    VideoSegmentClassification,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


class Cosmos3VideoSegmentClassification(VideoClassificationModel):
    def __init__(self, reasoner: Cosmos3EdgeReasoner):
        self._reasoner = reasoner

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "Cosmos3VideoSegmentClassification":
        reasoner = Cosmos3EdgeReasoner.from_pretrained(model_name_or_path, **kwargs)
        return cls(reasoner=reasoner)

    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassification]:
        segments = self._reasoner.temporal_localization(
            frames=frames,
            class_names=class_names,
            input_color_format="rgb",
            fps=fps,
            **kwargs,
        )
        return [
            VideoSegmentClassification(
                start_frame_idx=segment["start_frame_idx"],
                end_frame_idx=segment["end_frame_idx"],
                class_name=segment["class"],
            )
            for segment in segments
        ]
