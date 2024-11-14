"""
This test module requires Google AI API key passed via env variable WORKFLOWS_TEST_GOOGLE_API_KEY.
This is supposed to be used only locally, as that would be too much of a cost in CI
"""

import os

import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

GOOGLE_API_KEY = os.getenv("WORKFLOWS_TEST_GOOGLE_API_KEY")

UNCONSTRAINED_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "unconstrained",
            "prompt": "Give me dominant color of the image",
            "api_key": "$inputs.api_key",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.gemini.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Prompting Google's Gemini with arbitrary prompt",
    use_case_description="""
In this example, Google's Gemini model is prompted with arbitrary text from user 
    """,
    workflow_definition=UNCONSTRAINED_WORKFLOW,
    workflow_name_in_app="gemini-arbitrary-prompt",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_unconstrained_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=UNCONSTRAINED_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image, license_plate_image],
            "api_key": GOOGLE_API_KEY,
            "prompt": "What is the topic of the image?",
        }
    )

    # then
    assert len(result) == 2, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert set(result[1].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["result"], str) and len(result[0]["result"]) > 0
    ), "Expected non-empty string generated"
    assert (
        isinstance(result[1]["result"], str) and len(result[1]["result"]) > 0
    ), "Expected non-empty string generated"


OCR_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "ocr",
            "api_key": "$inputs.api_key",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.gemini.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as OCR model",
    use_case_description="""
In this example, Google's Gemini model is used as OCR system. User just points task type and do not need to provide
any prompt.
    """,
    workflow_definition=OCR_WORKFLOW,
    workflow_name_in_app="gemini-ocr",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_ocr_prompt(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OCR_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
            "api_key": GOOGLE_API_KEY,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["result"], str) and len(result[0]["result"]) > 0
    ), "Expected non-empty string generated"


VQA_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "prompt"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "visual-question-answering",
            "prompt": "$inputs.prompt",
            "api_key": "$inputs.api_key",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.gemini.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as Visual Question Answering system",
    use_case_description="""
In this example, Google's Gemini model is used as VQA system. User provides question via prompt.
    """,
    workflow_definition=VQA_WORKFLOW,
    workflow_name_in_app="gemini-vqa",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_vqa_prompt(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=VQA_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
            "api_key": GOOGLE_API_KEY,
            "prompt": "What are the brands of the cars?",
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["result"], str) and len(result[0]["result"]) > 0
    ), "Expected non-empty string generated"


CAPTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "caption",
            "api_key": "$inputs.api_key",
            "temperature": 1.0,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.gemini.output",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as Image Captioning system",
    use_case_description="""
In this example, Google's Gemini model is used as Image Captioning system.
    """,
    workflow_definition=CAPTION_WORKFLOW,
    workflow_name_in_app="gemini-captioning",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_captioning_prompt(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CAPTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image],
            "api_key": GOOGLE_API_KEY,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["result"], str) and len(result[0]["result"]) > 0
    ), "Expected non-empty string generated"


CLASSIFICATION_WORKFLOW_LEGACY_PARSER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v1",
            "name": "parser",
            "image": "$inputs.image",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "top_class",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
            "data": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "top_class",
            "selector": "$steps.top_class.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.*",
        },
    ],
}


@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_multi_class_classifier_prompt_and_legacy_parser(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW_LEGACY_PARSER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "top_class",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["gemini_result"], str)
        and len(result[0]["gemini_result"]) > 0
    ), "Expected non-empty string generated"
    assert result[0]["top_class"] == "dog"
    assert result[0]["parsed_prediction"]["error_status"] is False


CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v2",
            "name": "parser",
            "image": "$inputs.image",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "top_class",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
            "data": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "top_class",
            "selector": "$steps.top_class.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.*",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as multi-class classifier",
    use_case_description="""
In this example, Google's Gemini model is used as classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.
    """,
    workflow_definition=CLASSIFICATION_WORKFLOW,
    workflow_name_in_app="gemini-multi-class-classifier",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_multi_class_classifier_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "top_class",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert (
        isinstance(result[0]["gemini_result"], str)
        and len(result[0]["gemini_result"]) > 0
    ), "Expected non-empty string generated"
    assert result[0]["top_class"] == "dog"
    assert result[0]["parsed_prediction"]["error_status"] is False


MULTI_LABEL_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "multi-label-classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v2",
            "name": "parser",
            "image": "$inputs.image",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "top_class",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
            "data": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.top_class.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.*",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as multi-label classifier",
    use_case_description="""
In this example, Google's Gemini model is used as multi-label classifier. Output from the model is parsed by
special `roboflow_core/vlm_as_classifier@v2` block which turns model output text into
full-blown prediction, which can later be used by other blocks compatible with 
classification predictions - in this case we extract top-class property.
    """,
    workflow_definition=MULTI_LABEL_CLASSIFICATION_WORKFLOW,
    workflow_name_in_app="gemini-multi-label-classifier",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_multi_label_classifier_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=MULTI_LABEL_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "result",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert result[0]["result"] == ["dog"]
    assert result[0]["parsed_prediction"]["error_status"] is False


STRUCTURED_PROMPTING_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "structured-answering",
            "output_structure": {
                "dogs_count": "count of dogs instances in the image",
                "cats_count": "count of cats instances in the image",
            },
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/json_parser@v1",
            "name": "parser",
            "raw_json": "$steps.gemini.output",
            "expected_fields": ["dogs_count", "cats_count"],
        },
        {
            "type": "roboflow_core/property_definition@v1",
            "name": "property_definition",
            "operations": [{"type": "ToString"}],
            "data": "$steps.parser.dogs_count",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "result",
            "selector": "$steps.property_definition.output",
        }
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini to provide structured JSON",
    use_case_description="""
In this example, Google's Gemini model is expected to provide structured output in JSON, which can later be
parsed by dedicated `roboflow_core/json_parser@v1` block which transforms string into dictionary 
and expose it's keys to other blocks for further processing. In this case, parsed output is
transformed using `roboflow_core/property_definition@v1` block.
    """,
    workflow_definition=STRUCTURED_PROMPTING_WORKFLOW,
    workflow_name_in_app="gemini-structured-prompting",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_structured_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=STRUCTURED_PROMPTING_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {"result"}, "Expected all outputs to be delivered"
    assert result[0]["result"] == "2"


OBJECT_DETECTION_WORKFLOW_LEGACY_PARSER = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v1",
            "name": "parser",
            "vlm_output": "$steps.gemini.output",
            "image": "$inputs.image",
            "classes": "$steps.gemini.classes",
            "model_type": "google-gemini",
            "task_type": "object-detection",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.predictions",
        },
    ],
}


@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_object_detection_prompt_and_legacy_parser(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW_LEGACY_PARSER,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert result[0]["parsed_prediction"].data["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected 2 dogs to be detected"


OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {"type": "WorkflowParameter", "name": "classes"},
    ],
    "steps": [
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$inputs.image",
            "task_type": "object-detection",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_detector@v2",
            "name": "parser",
            "vlm_output": "$steps.gemini.output",
            "image": "$inputs.image",
            "classes": "$steps.gemini.classes",
            "model_type": "google-gemini",
            "task_type": "object-detection",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "gemini_result",
            "selector": "$steps.gemini.output",
        },
        {
            "type": "JsonField",
            "name": "parsed_prediction",
            "selector": "$steps.parser.predictions",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as object-detection model",
    use_case_description="""
In this example, Google's Gemini model is expected to provide output, which can later be
parsed by dedicated `roboflow_core/vlm_as_detector@v1` block which transforms string into `sv.Detections`, 
which can later be used by other blocks processing object-detection predictions.
    """,
    workflow_definition=OBJECT_DETECTION_WORKFLOW,
    workflow_name_in_app="gemini-object-detection",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_object_detection_prompt(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["cat", "dog"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "gemini_result",
        "parsed_prediction",
    }, "Expected all outputs to be delivered"
    assert result[0]["parsed_prediction"].data["class_name"].tolist() == [
        "dog",
        "dog",
    ], "Expected 2 dogs to be detected"


VLM_AS_SECONDARY_CLASSIFIER_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "api_key"},
        {
            "type": "WorkflowParameter",
            "name": "classes",
            "default_value": [
                "russell-terrier",
                "wirehaired-pointing-griffon",
                "beagle",
            ],
        },
    ],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "roboflow_core/google_gemini@v1",
            "name": "gemini",
            "images": "$steps.cropping.crops",
            "task_type": "classification",
            "classes": "$inputs.classes",
            "api_key": "$inputs.api_key",
        },
        {
            "type": "roboflow_core/vlm_as_classifier@v2",
            "name": "parser",
            "image": "$steps.cropping.crops",
            "vlm_output": "$steps.gemini.output",
            "classes": "$steps.gemini.classes",
        },
        {
            "type": "roboflow_core/detections_classes_replacement@v1",
            "name": "classes_replacement",
            "object_detection_predictions": "$steps.general_detection.predictions",
            "classification_predictions": "$steps.parser.predictions",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.classes_replacement.predictions",
        },
    ],
}


@add_to_workflows_gallery(
    category="Workflows with Visual Language Models",
    use_case_title="Using Google's Gemini as secondary classifier",
    use_case_description="""
In this example, Google's Gemini model is used as secondary classifier - first, YOLO model
detects dogs, then for each dog we run classification with VLM and at the end we replace 
detections classes to have bounding boxes with dogs breeds labels.

Breeds that we classify: `russell-terrier`, `wirehaired-pointing-griffon`, `beagle`
    """,
    workflow_definition=VLM_AS_SECONDARY_CLASSIFIER_WORKFLOW,
    workflow_name_in_app="gemini-secondary-classifier",
)
@pytest.mark.skipif(
    condition=GOOGLE_API_KEY is None, reason="Google API key not provided"
)
def test_workflow_with_secondary_classifier(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=VLM_AS_SECONDARY_CLASSIFIER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [dogs_image],
            "api_key": GOOGLE_API_KEY,
            "classes": ["russell-terrier", "wirehaired-pointing-griffon", "beagle"],
        }
    )

    # then
    assert len(result) == 1, "Single image given, expected single output"
    assert set(result[0].keys()) == {
        "predictions",
    }, "Expected all outputs to be delivered"
    assert "dog" not in set(
        result[0]["predictions"].data["class_name"].tolist()
    ), "Expected classes to be substituted"
