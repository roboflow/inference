import itertools
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from inference.core.managers.base import ModelManager
from inference.core.utils.image_utils import ImageType, load_image
from inference.enterprise.deployments.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.deployments.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.deployments.complier.utils import construct_step_selector
from inference.enterprise.deployments.entities.steps import Condition, Crop, Operator

OPERATORS = {
    Operator.EQUAL: lambda a, b: a == b,
    Operator.NOT_EQUAL: lambda a, b: a != b,
    Operator.LOWER_THAN: lambda a, b: a < b,
    Operator.GREATER_THAN: lambda a, b: a > b,
    Operator.LOWER_OR_EQUAL_THAN: lambda a, b: a <= b,
    Operator.GREATER_OR_EQUAL_THAN: lambda a, b: a >= b,
    Operator.IN: lambda a, b: a in b,
}


async def run_crop_step(
    step: Crop,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    detections = resolve_parameter(
        selector_or_value=step.detections,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if issubclass(type(image), list):
        decoded_image = [load_image(e) for e in image]
        decoded_image = [
            i[0] if i[1] is True else i[0][:, :, ::-1] for i in decoded_image
        ]
        crops = list(
            itertools.chain.from_iterable(
                crop_image(image=i, detections=d)
                for i, d in zip(decoded_image, detections)
            )
        )
    else:
        decoded_image, is_bgr = load_image(image)
        if not is_bgr:
            decoded_image = decoded_image[:, :, ::-1]
        crops = crop_image(image=decoded_image, detections=detections)
    parent_ids = [c["parent_id"] for c in crops]
    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "crops": crops,
        "parent_id": parent_ids,
    }
    return None, outputs_lookup


def crop_image(
    image: np.ndarray,
    detections: List[dict],
) -> List[Dict[str, Union[str, np.ndarray]]]:
    crops = []
    for detection in detections:
        x_min = round(detection["x"] - detection["width"] / 2)
        y_min = round(detection["y"] - detection["height"] / 2)
        x_max = round(x_min + detection["width"])
        y_max = round(y_min + detection["height"])
        cropped_image = image[y_min:y_max, x_min:x_max]
        crops.append(
            {
                "type": ImageType.NUMPY_OBJECT.value,
                "value": cropped_image,
                "parent_id": detection["detection_id"],
            }
        )
    return crops


async def run_condition_step(
    step: Condition,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    left_value = resolve_parameter(
        selector_or_value=step.left,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    right_value = resolve_parameter(
        selector_or_value=step.right,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    evaluation_result = OPERATORS[step.operator](left_value, right_value)
    next_step = step.step_if_true if evaluation_result else step.step_if_false
    return next_step, outputs_lookup
