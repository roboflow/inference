"""Tensor-input sibling of the action recognition workflow block."""

from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import torch

from inference.core.env import WORKFLOWS_IMAGE_TENSOR_DEVICE
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
    BlockManifest,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1 as _NumpyActionRecognitionModelBlockV1,
)
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_KEY,
    CLASSIFICATION_STYLE_MODEL,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)


class ActionRecognitionModelBlockV1(_NumpyActionRecognitionModelBlockV1):
    def _extract_frame(self, image: WorkflowImageData):
        if image.is_tensor_materialised():
            frame = image.tensor_image
            if frame.dim() != 3 or frame.shape[0] != 3:
                raise ValueError(
                    "Action Recognition Model expects a CHW RGB frame tensor."
                )
            return frame
        return np.ascontiguousarray(image.numpy_image[:, :, ::-1])

    def _build_window_classes(
        self,
        bookkeeping: Any,
        image: WorkflowImageData,
        id_vocabulary: Optional[List[str]],
    ) -> MultiLabelClassificationPrediction:
        window_class_names = list(bookkeeping.window_class_names)
        if id_vocabulary is not None:
            class_names: Dict[int, str] = {
                index: name for index, name in enumerate(id_vocabulary)
            }
            name_to_id = {name: index for index, name in class_names.items()}
            predicted_class_ids = [
                name_to_id[name]
                for name in window_class_names
                if name in name_to_id
            ]
            confidence = [0.0] * len(id_vocabulary)
            for class_id in predicted_class_ids:
                confidence[class_id] = 1.0
        else:
            class_names = {
                index: name for index, name in enumerate(window_class_names)
            }
            predicted_class_ids = list(range(len(window_class_names)))
            confidence = [1.0] * len(window_class_names)

        if not window_class_names:
            confidence = []

        height, width = image._read_shape_without_materialization()
        return MultiLabelClassificationPrediction(
            class_ids=torch.as_tensor(
                predicted_class_ids,
                dtype=torch.long,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            ),
            confidence=torch.as_tensor(
                confidence,
                dtype=torch.float32,
                device=WORKFLOWS_IMAGE_TENSOR_DEVICE,
            ),
            image_metadata={
                CLASS_NAMES_KEY: class_names,
                CLASSIFICATION_STYLE_KEY: CLASSIFICATION_STYLE_MODEL,
                PREDICTION_TYPE_KEY: "classification",
                IMAGE_DIMENSIONS_KEY: [height, width],
                INFERENCE_ID_KEY: str(uuid4()),
                PARENT_ID_KEY: image.parent_metadata.parent_id,
                ROOT_PARENT_ID_KEY: image.workflow_root_ancestor_metadata.parent_id,
            },
        )
