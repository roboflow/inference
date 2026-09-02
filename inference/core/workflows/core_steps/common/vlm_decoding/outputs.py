"""Output definitions and task dispatch for VLM blocks that decode output.

A VLM block declares one ``predictions`` output whose kind is narrowed to
whatever the selected task actually produces. The engine requires a step's
returned keys to exactly equal its declared actual outputs (see
``BatchStepCache.register_outputs``), so ``predictions`` is always declared
even for tasks that decode nothing - it is simply ``None`` at runtime.
"""

from typing import Any, List, Optional, Tuple

from inference.core.logger import logger
from inference.core.workflows.core_steps.common.vlm_decoding.classification import (
    decode_classification,
)
from inference.core.workflows.core_steps.common.vlm_decoding.detections import (
    decode_object_detections,
)
from inference.core.workflows.core_steps.common.vlm_decoding.tensor_native import (
    to_tensor_native_predictions,
)
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INFERENCE_ID_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
)

DETECTION_TASKS = {"object-detection"}
CLASSIFICATION_TASKS = {"classification", "multi-label-classification"}


def describe_vlm_prediction_outputs() -> List[OutputDefinition]:
    """Declare the manifest-level outputs of a decoding VLM block.

    Returns:
        The three shared outputs, with ``predictions`` typed as the union of
        every kind a task may produce.
    """
    return [
        OutputDefinition(
            name="predictions",
            kind=[OBJECT_DETECTION_PREDICTION_KIND, CLASSIFICATION_PREDICTION_KIND],
        ),
        OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        OutputDefinition(name="inference_id", kind=[INFERENCE_ID_KIND]),
    ]


def actual_vlm_prediction_outputs(task_type: str) -> List[OutputDefinition]:
    """Declare the outputs of a decoding VLM block for one selected task.

    Args:
        task_type: Task the block is configured to run.

    Returns:
        The three shared outputs, with ``predictions`` narrowed to the kind
        the task produces. Tasks that decode nothing keep the union kind and
        return ``None``.
    """
    if task_type in DETECTION_TASKS:
        prediction_kind = [OBJECT_DETECTION_PREDICTION_KIND]
    elif task_type in CLASSIFICATION_TASKS:
        prediction_kind = [CLASSIFICATION_PREDICTION_KIND]
    else:
        prediction_kind = [
            OBJECT_DETECTION_PREDICTION_KIND,
            CLASSIFICATION_PREDICTION_KIND,
        ]
    return [
        OutputDefinition(name="predictions", kind=prediction_kind),
        OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
        OutputDefinition(name="inference_id", kind=[INFERENCE_ID_KIND]),
    ]


def decode_vlm_output(
    task_type: str,
    raw_output: str,
    image: WorkflowImageData,
    classes: Optional[List[str]],
    inference_id: str,
    box_format: Optional[str] = None,
    upload_width: Optional[int] = None,
    upload_height: Optional[int] = None,
) -> Tuple[bool, Any]:
    """Decode a VLM answer according to the task the block ran.

    Args:
        task_type: Task the block is configured to run.
        raw_output: Raw string produced by the model.
        image: Workflow image the prediction refers to.
        classes: Class names, required for both decoding task families.
        inference_id: Identifier attached to the prediction.
        box_format: Registered box coordinate format, detection tasks only.
        upload_width: Width of the image as uploaded, for absolute formats.
        upload_height: Height of the image as uploaded, for absolute formats.

    Returns:
        Tuple of ``(error_status, predictions)``. Tasks outside the decoding
        families return ``(False, None)``. Under
        ``ENABLE_TENSOR_DATA_REPRESENTATION`` ``predictions`` is the
        tensor-native carrier of its kind instead of ``sv.Detections`` / a
        dict - the single conversion point for every VLM block, which is why
        none of them needs a ``_tensor`` sibling.
    """
    if task_type in DETECTION_TASKS:
        if box_format is None or classes is None:
            logger.warning(
                "Could not decode VLM object-detection output for task %s - "
                "a box format and a class list are both required.",
                task_type,
            )
            return True, None
        error_status, predictions = decode_object_detections(
            raw_output=raw_output,
            box_format=box_format,
            image=image,
            classes=classes,
            inference_id=inference_id,
            upload_width=upload_width,
            upload_height=upload_height,
        )
    elif task_type in CLASSIFICATION_TASKS:
        if classes is None:
            logger.warning(
                "Could not decode VLM classification output for task %s - "
                "a class list is required.",
                task_type,
            )
            return True, None
        error_status, predictions = decode_classification(
            raw_output=raw_output,
            image=image,
            classes=classes,
            inference_id=inference_id,
        )
    else:
        return False, None
    if predictions is None:
        return error_status, None
    return error_status, to_tensor_native_predictions(
        predictions=predictions,
        image=image,
        classes=classes,
    )
