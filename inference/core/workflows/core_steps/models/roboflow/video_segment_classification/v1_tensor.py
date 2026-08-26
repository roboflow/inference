"""Tensor-input sibling of the video classification workflow block."""

import numpy as np

from inference.core.workflows.core_steps.models.roboflow.video_segment_classification.v1 import (
    BlockManifest,
)
from inference.core.workflows.core_steps.models.roboflow.video_segment_classification.v1 import (
    VideoSegmentClassificationModelBlockV1 as _NumpyVideoSegmentClassificationModelBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData


class VideoSegmentClassificationModelBlockV1(_NumpyVideoSegmentClassificationModelBlockV1):
    def _extract_frame(self, image: WorkflowImageData):
        if image.is_tensor_materialised():
            frame = image.tensor_image
            if frame.dim() != 3 or frame.shape[0] != 3:
                raise ValueError(
                    "Video Segment Classification Model expects a CHW RGB frame tensor."
                )
            return frame
        return np.ascontiguousarray(image.numpy_image[:, :, ::-1])
