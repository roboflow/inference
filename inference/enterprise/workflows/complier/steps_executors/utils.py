from typing import Any, Dict, Generator, Iterable, List, TypeVar, Union

import numpy as np

from inference.enterprise.workflows.complier.steps_executors.types import OutputsLookup
from inference.enterprise.workflows.complier.utils import (
    get_step_selector_from_its_output,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.workflows.entities.steps import (
    LMM,
    AbsoluteStaticCrop,
    ActiveLearningDataCollector,
    ClipComparison,
    Crop,
    LMMForClassification,
    OCRModel,
    RelativeStaticCrop,
    RoboflowModel,
    YoloWorld,
)
from inference.enterprise.workflows.entities.validators import (
    get_last_selector_chunk,
    is_selector,
)
from inference.enterprise.workflows.errors import ExecutionGraphError

T = TypeVar("T")


def get_image(
    step: Union[
        RoboflowModel,
        OCRModel,
        Crop,
        AbsoluteStaticCrop,
        RelativeStaticCrop,
        ClipComparison,
        ActiveLearningDataCollector,
        YoloWorld,
        LMM,
        LMMForClassification,
    ],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> List[Dict[str, Union[str, np.ndarray]]]:
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


def make_batches(
    iterable: Iterable[T], batch_size: int
) -> Generator[List[T], None, None]:
    batch_size = max(batch_size, 1)
    batch = []
    for element in iterable:
        batch.append(element)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if len(batch) > 0:
        yield batch
