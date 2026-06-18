from collections import Counter
from copy import deepcopy
from typing import Dict

import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.query_language.errors import (
    InvalidInputTypeError,
    OperationError,
    UndeclaredSymbolError,
)
from inference.core.workflows.execution_engine.constants import CLASS_NAMES_KEY
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.object_detection import (
    Detections as NativeDetections,
)
from tests.workflows.integration_tests.execution.tensor_input_utils import (
    numpy_image_as_tensor,
)
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

# Under ENABLE_TENSOR_DATA_REPRESENTATION the DetectionsRename UQL op emits a native
# inference_models.Detections instead of sv.Detections. Native Detections is a plain
# dataclass: it has no __getitem__ (so `det["class_name"]` raises TypeError) and class
# names live in image_metadata[CLASS_NAMES_KEY] rather than data["class_name"]. The
# upstream tensor ObjectDetectionModel/NMS also returns detections in a different
# positional order than the numpy path, so the sv-shaped tests below (which assert an
# exact ordering via subscript access) run only with the flag off; an equivalent
# *_tensor_native test asserts the same order-independent semantic result against the
# native carrier with the flag on.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections output + exact ordering; native carrier with different "
    "upstream ordering under ENABLE_TENSOR_DATA_REPRESENTATION — see the "
    "*_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)


def build_class_remapping_workflow_definition(
    class_map: Dict[str, str],
    strict: bool,
) -> dict:
    return {
        "version": "1.0",
        "inputs": [
            {"type": "WorkflowImage", "name": "image"},
            {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
        ],
        "steps": [
            {
                "type": "ObjectDetectionModel",
                "name": "model",
                "image": "$inputs.image",
                "model_id": "yolov8n-640",
                "confidence": "$inputs.confidence",
            },
            {
                "type": "DetectionsTransformation",
                "name": "class_rename",
                "predictions": "$steps.model.predictions",
                "operations": [
                    {
                        "type": "DetectionsRename",
                        "strict": strict,
                        "class_map": class_map,
                    }
                ],
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "original_predictions",
                "selector": "$steps.model.predictions",
            },
            {
                "type": "JsonField",
                "name": "renamed_predictions",
                "selector": "$steps.class_rename.predictions",
            },
        ],
    }


@add_to_workflows_gallery(
    category="Workflows with data transformations",
    use_case_title="Workflow with detections class remapping",
    use_case_description="""
This workflow presents how to use Detections Transformation block that is going to 
change the name of the following classes: `apple`, `banana` into `fruit`.

In this example, we use non-strict mapping, causing new class `fruit` to be added to
pool of classes - you can see that if `banana` or `apple` is detected, the
class name changes to `fruit` and class id is 1024.

You can test the execution submitting image like 
[this](https://www.pexels.com/photo/four-trays-of-varieties-of-fruits-1300975/).
    """,
    workflow_definition=build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit"},
        strict=False,
    ),
    workflow_name_in_app="detections-class-remapping",
)
@_NUMPY_ONLY
def test_class_rename_workflow_with_non_strict_mapping(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit"},
        strict=False,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"

    if not USE_INFERENCE_MODELS:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "orange",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            1024,
            1024,
            1024,
            49,
            1024,
        ], "Expected renamed set of class ids to be the same as when test was created"
    else:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "orange",
            "fruit",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            1024,
            1024,
            1024,
            49,
            1024,
            1024,
        ], "Expected renamed set of class ids to be the same as when test was created"
    assert len(result[0]["renamed_predictions"]) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions no to change"


@_NUMPY_ONLY
def test_class_rename_workflow_with_strict_mapping_when_all_classes_are_remapped(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit", "orange": "my-orange"},
        strict=True,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"

    if not USE_INFERENCE_MODELS:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "my-orange",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            0,
            0,
            0,
            1,
            0,
        ], "Expected renamed set of class ids to be the same as when test was created"
    else:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "my-orange",
            "fruit",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            0,
            0,
            0,
            1,
            0,
            0,
        ], "Expected renamed set of class ids to be the same as when test was created"
    assert len(result[0]["renamed_predictions"]) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions no to change"


def test_class_rename_workflow_with_strict_mapping_when_not_all_classes_are_remapped(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit"},
        strict=True,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(OperationError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": fruit_image,
                "model_id": "yolov8n-640",
            },
        )


WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
        {"type": "WorkflowParameter", "name": "class_map"},
        {"type": "WorkflowParameter", "name": "strict"},
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "model",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "DetectionsTransformation",
            "name": "class_rename",
            "predictions": "$steps.model.predictions",
            "operations": [
                {
                    "type": "DetectionsRename",
                    "strict": "strict",
                    "class_map": "class_map",
                }
            ],
            "operations_parameters": {
                "class_map": "$inputs.class_map",
                "strict": "$inputs.strict",
            },
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "original_predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "renamed_predictions",
            "selector": "$steps.class_rename.predictions",
        },
    ],
}


@_NUMPY_ONLY
def test_class_rename_workflow_when_mapping_is_parametrised(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
            "class_map": {"apple": "fruit", "banana": "fruit"},
            "strict": False,
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"

    if not USE_INFERENCE_MODELS:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "orange",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            1024,
            1024,
            1024,
            49,
            1024,
        ], "Expected renamed set of class ids to be the same as when test was created"
    else:
        assert result[0]["renamed_predictions"]["class_name"].tolist() == [
            "fruit",
            "fruit",
            "fruit",
            "orange",
            "fruit",
            "fruit",
        ], "Expected renamed set of classes to be the same as when test was created"
        assert result[0]["renamed_predictions"].class_id.tolist() == [
            1024,
            1024,
            1024,
            49,
            1024,
            1024,
        ], "Expected renamed set of class ids to be the same as when test was created"
    assert len(result[0]["renamed_predictions"]) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions no to change"


def test_class_rename_workflow_when_mapping_is_parametrised_with_invalid_value(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(InvalidInputTypeError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": fruit_image,
                "model_id": "yolov8n-640",
                "class_map": "INVALID",
                "strict": False,
            },
        )


def test_class_rename_workflow_when_mapping_is_not_passed_as_operation_parameter_leaving_undeclared_symbol(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    workflow_definition = deepcopy(WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME)
    del workflow_definition["steps"][1]["operations_parameters"]
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(UndeclaredSymbolError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": fruit_image,
                "model_id": "yolov8n-640",
                "class_map": "INVALID",
                "strict": False,
            },
        )


def _native_class_names(detections: NativeDetections) -> Counter:
    """Resolve per-detection class names from the native carrier.

    On native inference_models.Detections the class names live in
    image_metadata[CLASS_NAMES_KEY] keyed by class id (there is no data["class_name"]
    and no __getitem__). The renamed detections come back in a different positional
    order than the numpy path, so callers compare the resulting multiset rather than
    an exact ordering.
    """
    class_names_map = detections.image_metadata[CLASS_NAMES_KEY]
    return Counter(
        class_names_map[int(class_id)] for class_id in detections.class_id.tolist()
    )


@_TENSOR_ONLY
def test_class_rename_workflow_with_non_strict_mapping_tensor_native(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit"},
        strict=False,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Same semantic result as the numpy path (order-independent): 5 boxes renamed to
    # `fruit` with new non-strict id 1024, the single `orange` (id 49) kept as-is.
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {1024: 5, 49: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"


@_TENSOR_ONLY
def test_class_rename_workflow_with_strict_mapping_when_all_classes_are_remapped_tensor_native(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit", "orange": "my-orange"},
        strict=True,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Strict mapping reindexes the resulting classes from 0 (sorted): fruit -> 0,
    # my-orange -> 1. Same semantic result as the numpy path (order-independent).
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "my-orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {0: 5, 1: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"


@_TENSOR_ONLY
def test_class_rename_workflow_when_mapping_is_parametrised_tensor_native(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": fruit_image,
            "model_id": "yolov8n-640",
            "class_map": {"apple": "fruit", "banana": "fruit"},
            "strict": False,
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Same semantic result as the numpy path (order-independent): 5 boxes renamed to
    # `fruit` with new non-strict id 1024, the single `orange` (id 49) kept as-is.
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {1024: 5, 49: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"


@_TENSOR_ONLY
def test_class_rename_workflow_with_non_strict_mapping_with_tensor_input(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # Same as test_class_rename_workflow_with_non_strict_mapping_tensor_native, but the
    # image arrives ALREADY materialised as a CHW RGB device tensor
    # (is_tensor_materialised() == True), so the OD block runs its on-device tensor path.
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit"},
        strict=False,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": numpy_image_as_tensor(fruit_image),
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Same semantic result as the numpy path (order-independent): 5 boxes renamed to
    # `fruit` with new non-strict id 1024, the single `orange` (id 49) kept as-is.
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {1024: 5, 49: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"


@_TENSOR_ONLY
def test_class_rename_workflow_with_strict_mapping_when_all_classes_are_remapped_with_tensor_input(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # Same as
    # test_class_rename_workflow_with_strict_mapping_when_all_classes_are_remapped_tensor_native,
    # but the image arrives ALREADY materialised as a CHW RGB device tensor
    # (is_tensor_materialised() == True), so the OD block runs its on-device tensor path.
    workflow_definition = build_class_remapping_workflow_definition(
        class_map={"apple": "fruit", "banana": "fruit", "orange": "my-orange"},
        strict=True,
    )

    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=workflow_definition,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": numpy_image_as_tensor(fruit_image),
            "model_id": "yolov8n-640",
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Strict mapping reindexes the resulting classes from 0 (sorted): fruit -> 0,
    # my-orange -> 1. Same semantic result as the numpy path (order-independent).
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "my-orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {0: 5, 1: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"


@_TENSOR_ONLY
def test_class_rename_workflow_when_mapping_is_parametrised_with_tensor_input(
    model_manager: ModelManager,
    fruit_image: np.ndarray,
) -> None:
    # Same as test_class_rename_workflow_when_mapping_is_parametrised_tensor_native, but
    # the image arrives ALREADY materialised as a CHW RGB device tensor
    # (is_tensor_materialised() == True), so the OD block runs its on-device tensor path.
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_PARAMETRISED_DETECTIONS_RENAME,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": numpy_image_as_tensor(fruit_image),
            "model_id": "yolov8n-640",
            "class_map": {"apple": "fruit", "banana": "fruit"},
            "strict": False,
        },
    )

    # then
    assert isinstance(result, list), "Expected result to be list"
    assert len(result) == 1, "Single image provided - single output expected"
    renamed = result[0]["renamed_predictions"]
    assert isinstance(
        renamed, NativeDetections
    ), "Output must be native inference_models.Detections under tensor mode"
    # Same semantic result as the numpy path (order-independent): 5 boxes renamed to
    # `fruit` with new non-strict id 1024, the single `orange` (id 49) kept as-is.
    assert _native_class_names(renamed) == Counter(
        {"fruit": 5, "orange": 1}
    ), "Expected renamed class names multiset to match the numpy baseline"
    assert Counter(renamed.class_id.tolist()) == Counter(
        {1024: 5, 49: 1}
    ), "Expected renamed class id multiset to match the numpy baseline"
    assert len(renamed) == len(
        result[0]["original_predictions"]
    ), "Expected length of predictions not to change"
