import copy
import json
import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from tests.workflows.integration_tests.execution.conftest import bool_env
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.4},
    ],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
            "confidence": "$inputs.confidence",
        },
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-classification",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Florence 2 - grounded classification",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image, which are later classified by Florence 2 model. With Workflows it is possible 
to pass `grounding_detection` as an input for all of the tasks named `detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.
    """,
    workflow_definition=FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION,
    workflow_name_in_app="florence-2-detection-grounded-classification",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_grounded_classification(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"

    assert json.loads(result[0]["model_predictions"]["raw_output"]).startswith(
        "dog"
    ), "Expected dog to be output by florence2"


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_grounded_classification_when_no_grounding_available(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_CLASSIFICATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "confidence": 0.99,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"
    assert result[0]["model_predictions"]["raw_output"] is None, "Expected no output"
    assert result[0]["model_predictions"]["parsed_output"] is None, "Expected no output"
    assert result[0]["model_predictions"]["classes"] is None, "Expected no output"


FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-instance-segmentation",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Florence 2 - grounded segmentation",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image and run segmentation of selected region with Florence 2. With Workflows it is 
possible to pass `grounding_detection` as an input for all of the tasks named 
`detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.
    """,
    workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
    workflow_name_in_app="florence-2-detection-grounded-segmentation",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_grounded_instance_segmentation(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result[0]["model_predictions"]["parsed_output"]["polygons"], list
    ), "Expected list of polygons in the output"


FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_GROUNDED_BY_INPUT_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "bounding_box"},
    ],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-instance-segmentation",
            "grounding_detection": "$inputs.bounding_box",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_instance_segmentation_grounded_by_input(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
            "bounding_box": [],
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"
    assert isinstance(
        result[0]["model_predictions"]["parsed_output"]["polygons"], list
    ), "Expected list of polygons in the output"


FLORENCE2_GROUNDED_CAPTION_WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/roboflow_object_detection_model@v2",
            "name": "model_1",
            "images": "$inputs.image",
            "model_id": "yolov8n-640",
        },
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "detection-grounded-caption",
            "grounding_detection": "$steps.model_1.predictions",
            "grounding_selection_mode": "most-confident",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Florence 2 - grounded captioning",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use object detection model to find regions of interest in the 
input image and run captioning of selected region with Florence 2. With Workflows it is 
possible to pass `grounding_detection` as an input for all of the tasks named 
`detection-grounded-*`.

Grounding detection can either be input parameter or output of detection model. If the 
latter is true, one should choose `grounding_selection_mode` - as Florence do only support 
a single bounding box as grounding - when multiple detections can be provided, block
will select one based on parameter.
    """,
    workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
    workflow_name_in_app="florence-2-detection-grounded-caption",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_grounded_caption(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE2_GROUNDED_CAPTION_WORKFLOW_DEFINITION,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "model_predictions",
    }, "Expected all declared outputs to be delivered"

    assert json.loads(result[0]["model_predictions"]["raw_output"]).startswith(
        "dog"
    ), "Expected dog to be output by florence2"


FLORENCE_OBJECT_DETECTION_WORKFLOW_LEGACY_PARSER = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "open-vocabulary-object-detection",
            "classes": "$inputs.classes",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v1",
            "name": "vlm_as_detector",
            "image": "$inputs.image",
            "vlm_output": "$steps.model.raw_output",
            "classes": "$steps.model.classes",
            "model_type": "florence-2",
            "task_type": "open-vocabulary-object-detection",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.vlm_as_detector.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.vlm_as_detector.predictions",
        },
        {
            "type": "JsonField",
            "name": "bounding_box_visualization",
            "coordinates_system": "own",
            "selector": "$steps.bounding_box_visualization.image",
        },
    ],
}


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_object_detection_when_legacy_parser_is_in_use(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE_OBJECT_DETECTION_WORKFLOW_LEGACY_PARSER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image, "classes": ["dog"]}
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
        "bounding_box_visualization",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["predictions"]) == 2, "Expected two predictions"
    assert result[0]["predictions"].data["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected two dogs to be found"


FLORENCE_OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "open-vocabulary-object-detection",
            "classes": "$inputs.classes",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "vlm_as_detector",
            "image": "$inputs.image",
            "vlm_output": "$steps.model.raw_output",
            "classes": "$steps.model.classes",
            "model_type": "florence-2",
            "task_type": "open-vocabulary-object-detection",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.vlm_as_detector.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.vlm_as_detector.predictions",
        },
        {
            "type": "JsonField",
            "name": "bounding_box_visualization",
            "coordinates_system": "own",
            "selector": "$steps.bounding_box_visualization.image",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Florence 2 - object detection",
    use_case_description="""
**THIS EXAMPLE CAN ONLY BE RUN LOCALLY OR USING DEDICATED DEPLOYMENT**

In this example, we use Florence 2 as zero-shot object detection model, specifically 
performing open-vocabulary detection. Input parameter `classes` can be used to
provide list of objects that model should find. Beware that Florence 2 is prone to 
seek for all of the classes provided in your list - so if you select class which is not
visible in the image, you can expect either big bounding box covering whole image, 
or multiple bounding boxes over one of detected instance, with auxiliary boxes
providing not meaningful labels for all of the objects you specified in class list.
    """,
    workflow_definition=FLORENCE2_GROUNDED_INSTANCE_SEGMENTATION_WORKFLOW_DEFINITION,
    workflow_name_in_app="florence-2-detection-grounded-caption",
)
@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
def test_florence2_object_detection(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=FLORENCE_OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image": dogs_image, "classes": ["dog"]}
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert set(result[0].keys()) == {
        "predictions",
        "bounding_box_visualization",
    }, "Expected all declared outputs to be delivered"
    assert len(result[0]["predictions"]) == 2, "Expected two predictions"
    assert result[0]["predictions"].data["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected two dogs to be found"


FLORENCE_OD_TASK_TYPES = [
    "object-detection",
    "open-vocabulary-object-detection",
    "object-detection-and-caption",
    "phrase-grounded-object-detection",
    "region-proposal",
    "ocr-with-text-detection",
]


FLORENCE_VLM_AS_DET_VISUALIZE_DEF = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/florence_2@v1",
            "name": "model",
            "images": "$inputs.image",
            "task_type": "object-detection",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "vlm_as_detector",
            "image": "$inputs.image",
            "vlm_output": "$steps.model.raw_output",
            "classes": "$steps.model.classes",
            "model_type": "florence-2",
            "task_type": "object-detection",
        },
        {
            "type": "roboflow_core/bounding_box_visualization@v1",
            "name": "bounding_box_visualization",
            "image": "$inputs.image",
            "predictions": "$steps.vlm_as_detector.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_predictions",
            "coordinates_system": "own",
            "selector": "$steps.model.*",
        },
        {
            "type": "JsonField",
            "name": "vlm_as_detector",
            "coordinates_system": "own",
            "selector": "$steps.vlm_as_detector.*",
        },
        {
            "type": "JsonField",
            "name": "bounding_box_visualization",
            "coordinates_system": "own",
            "selector": "$steps.bounding_box_visualization.image",
        },
    ],
}


def make_visualize_workflow(task_type):
    wf_def = copy.deepcopy(FLORENCE_VLM_AS_DET_VISUALIZE_DEF)
    if task_type == "phrase-grounded-object-detection":
        wf_def["steps"][0]["prompt"] = "dog"
    elif task_type == "open-vocabulary-object-detection":
        wf_def["steps"][0]["classes"] = ["dog"]
    wf_def["steps"][0]["task_type"] = task_type
    wf_def["steps"][1]["task_type"] = task_type
    return wf_def


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_FLORENCE2_TEST", True)), reason="Skipping Florence 2 test"
)
@pytest.mark.parametrize("task_type", FLORENCE_OD_TASK_TYPES)
def test_florence_visualization_with_vlm_as_detector_all_variations(
    task_type,
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=make_visualize_workflow(task_type),
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output for one input image"
    assert isinstance(result[0]["bounding_box_visualization"], WorkflowImageData)
