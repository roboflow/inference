from typing import List, Optional, Type, Union

from inference_models.models.base.classification import (
    ClassificationPrediction,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.object_detection import Detections as TensorDetections

from inference.core.workflows.core_steps.common.to_supervision import (
    classification_prediction_to_dict_per_image,
    multi_label_classification_to_dict,
    sv_detections_to_inference_models_detections,
    to_supervision_with_metadata,
)
from inference.core.workflows.core_steps.fusion.detections_classes_replacement.v1 import (
    BlockManifest,
    DetectionsClassesReplacementBlockV1 as _NumpyBlock,
)
from inference.core.workflows.execution_engine.entities.base import Batch
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

# Manifest re-exported verbatim from v1 — class name shared via loader swap.


class DetectionsClassesReplacementBlockV1(WorkflowBlock):
    """Tensor-mode sibling that wraps the numpy implementation.

    The classes-replacement logic is heavily sv.Detections-shaped (matches
    detection_id ↔ parent_id, replaces class arrays in .data, regenerates
    detection_ids). Reimplementing it natively on inference_models tensors
    would be a substantial port. Pragmatic vibe-mode path: convert inputs
    at the boundary, delegate to the numpy logic, convert outputs back to
    inference_models native.

    The materialisation cost is paid by this block; downstream tensor
    consumers receive `inference_models.Detections` and stay tensor-native
    from here on.
    """

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        object_detection_predictions,
        classification_predictions: Optional[
            Batch[Optional[Union[dict, str, List[str], ClassificationPrediction, MultiLabelClassificationPrediction]]]
        ],
        fallback_class_name: Optional[str],
        fallback_class_id: Optional[int],
    ) -> BlockResult:
        # Carry tensor metadata across the round-trip so downstream
        # consumers see the same inference_id / model_id / class_names as
        # the upstream producer attached.
        inherit_image_metadata = None
        if isinstance(object_detection_predictions, TensorDetections):
            inherit_image_metadata = object_detection_predictions.image_metadata
            sv_predictions = to_supervision_with_metadata(object_detection_predictions)
        else:
            sv_predictions = object_detection_predictions

        # Lower each tensor-classification value to the dict shape the numpy
        # logic expects. The block's existing dict path handles top-1 +
        # parent_id matching identically.
        lowered_classifications: List[Optional[Union[dict, str, List[str]]]] = []
        if classification_predictions is None:
            lowered = None
        else:
            for value in classification_predictions:
                lowered_classifications.append(_lower_classification_input(value))
            lowered = lowered_classifications

        numpy_result = _NumpyBlock().run(
            object_detection_predictions=sv_predictions,
            classification_predictions=lowered,
            fallback_class_name=fallback_class_name,
            fallback_class_id=fallback_class_id,
        )

        # numpy_result is typically {"predictions": sv.Detections | None}.
        # Convert back to inference_models native for tensor downstream.
        sv_out = numpy_result.get("predictions") if isinstance(numpy_result, dict) else None
        if sv_out is None:
            return {"predictions": None}
        return {
            "predictions": sv_detections_to_inference_models_detections(
                sv_out, inherit_image_metadata=inherit_image_metadata
            )
        }


def _lower_classification_input(value):
    """Translate inference_models classification types into the dict / str
    shape the numpy classes-replacement block already understands."""
    if value is None:
        return None
    if isinstance(value, ClassificationPrediction):
        # Single-label batch-shaped — should not appear here in the per-image
        # consumer path. Take the first slot defensively.
        dicts = classification_prediction_to_dict_per_image(value)
        return dicts[0] if dicts else None
    if isinstance(value, MultiLabelClassificationPrediction):
        return multi_label_classification_to_dict(value)
    return value  # str / List[str] / dict — passthrough
