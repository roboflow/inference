from typing import Any, Dict, Union

from inference.enterprise.deployments.complier.steps_executors.types import (
    OutputsLookup,
)
from inference.enterprise.deployments.complier.utils import (
    get_step_selector_from_its_output,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.deployments.entities.steps import (
    AbsoluteStaticCrop,
    ClipComparison,
    Crop,
    OCRModel,
    RelativeStaticCrop,
    RoboflowModel,
)
from inference.enterprise.deployments.entities.validators import (
    get_last_selector_chunk,
    is_selector,
)
from inference.enterprise.deployments.errors import ExecutionGraphError


def get_image(
    step: Union[
        RoboflowModel,
        OCRModel,
        Crop,
        AbsoluteStaticCrop,
        RelativeStaticCrop,
        ClipComparison,
    ],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> Any:
    if is_input_selector(selector_or_value=step.image):
        return runtime_parameters[get_last_selector_chunk(selector=step.image)]
    if is_step_output_selector(selector_or_value=step.image):
        step_selector = get_step_selector_from_its_output(
            step_output_selector=step.image
        )
        step_output = outputs_lookup[step_selector]
        return step_output[get_last_selector_chunk(selector=step.image)]
    raise ExecutionGraphError("Cannot find image")


def resolve_parameter(
    selector_or_value: Any,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> Any:
    if not is_selector(selector_or_value=selector_or_value):
        return selector_or_value
    if is_step_output_selector(selector_or_value=selector_or_value):
        step_selector = get_step_selector_from_its_output(
            step_output_selector=selector_or_value
        )
        step_output = outputs_lookup[step_selector]
        if issubclass(type(step_output), list):
            return [
                e[get_last_selector_chunk(selector=selector_or_value)]
                for e in step_output
            ]
        return step_output[get_last_selector_chunk(selector=selector_or_value)]
    return runtime_parameters[get_last_selector_chunk(selector=selector_or_value)]
