import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.errors import RuntimeInputError, StepExecutionError
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

CLIP_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image_1"},
        {"type": "InferenceImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_1",
            "data": "$inputs.image_1",
            "version": "RN50",
        },
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_2",
            "data": "$inputs.image_2",
            "version": "RN50",
        },
        {
            "type": "roboflow_core/cosine_similarity@v1",
            "name": "cosine_similarity",
            "embedding_1": "$steps.embedding_1.embedding",
            "embedding_2": "$steps.embedding_2.embedding",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "coordinates_system": "own",
            "selector": "$steps.cosine_similarity.similarity",
        },
        {
            "type": "JsonField",
            "name": "image_embeddings",
            "coordinates_system": "own",
            "selector": "$steps.embedding_1.embedding",
        },
    ],
}


@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with Embeddings",
    use_case_description="""
This Workflow shows how to use an embedding model to compare the
similarity of two images with each other.
    """,
    workflow_definition=CLIP_WORKFLOW,
    workflow_name_in_app="clip",
)
@pytest.mark.skip(reason="Known problem of race condition in execution engine")
def test_clip_embedding_model(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={"image_1": license_plate_image, "image_2": crowd_image}
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output"
    assert set(result[0].keys()) == {
        "similarity",
        "image_embeddings",
    }, "Expected all declared outputs to be delivered"
    assert (
        pytest.approx(result[0]["similarity"], 0.01) == 0.444
    ), "Expected similarity to be approximately the defined value"
    assert (
        len(result[0]["image_embeddings"]) == 1024
    ), "Expected image embedding to be of dimension 1024 for RN50 model"


CLIP_WORKFLOW_COSINE_SIMILARITY_CROSS_DATA_TYPE = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image_1"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_1",
            "data": "$inputs.image_1",
            "version": "RN50",
        },
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_2",
            "data": "$inputs.reference",
            "version": "RN50",
        },
        {
            "type": "roboflow_core/cosine_similarity@v1",
            "name": "cosine_similarity",
            "embedding_1": "$steps.embedding_1.embedding",
            "embedding_2": "$steps.embedding_2.embedding",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "coordinates_system": "own",
            "selector": "$steps.cosine_similarity.similarity",
        },
        {
            "type": "JsonField",
            "name": "image_embeddings",
            "coordinates_system": "own",
            "selector": "$steps.embedding_1.embedding",
        },
    ],
}


@pytest.mark.skip(reason="Known problem of race condition in execution engine")
def test_clip_embedding_model_on_batches_of_cross_type_data(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_WORKFLOW_COSINE_SIMILARITY_CROSS_DATA_TYPE,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image_1": [license_plate_image, crowd_image],
            "reference": "people",
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output"
    assert set(result[0].keys()) == {
        "similarity",
        "image_embeddings",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "similarity",
        "image_embeddings",
    }, "Expected all declared outputs to be delivered"
    assert (
        abs(result[0]["similarity"] - 0.13) < 0.02
    ), "Expected similarity to be approximately the defined value"
    assert (
        len(result[0]["image_embeddings"]) == 1024
    ), "Expected image embedding to be of dimension 1024 for RN50 model"
    assert (
        abs(result[1]["similarity"] - 0.15) < 0.02
    ), "Expected similarity to be approximately the defined value"
    assert (
        len(result[1]["image_embeddings"]) == 1024
    ), "Expected image embedding to be of dimension 1024 for RN50 model"


CLIP_WORKFLOW_COSINE_SIMILARITY_CROSS_DATA_TYPE_WITH_INVALID_LENGTH_OF_EMBEDDINGS = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image_1"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_1",
            "data": "$inputs.image_1",
            "version": "RN50",
        },
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding_2",
            "data": "$inputs.reference",
            "version": "RN50x4",
        },
        {
            "type": "roboflow_core/cosine_similarity@v1",
            "name": "cosine_similarity",
            "embedding_1": "$steps.embedding_1.embedding",
            "embedding_2": "$steps.embedding_2.embedding",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "coordinates_system": "own",
            "selector": "$steps.cosine_similarity.similarity",
        },
        {
            "type": "JsonField",
            "name": "image_embeddings",
            "coordinates_system": "own",
            "selector": "$steps.embedding_1.embedding",
        },
    ],
}


@pytest.mark.skip(reason="Known problem of race condition in execution engine")
def test_clip_embedding_model_on_batches_of_cross_type_data_with_different_embeddings_length(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_WORKFLOW_COSINE_SIMILARITY_CROSS_DATA_TYPE_WITH_INVALID_LENGTH_OF_EMBEDDINGS,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(StepExecutionError) as error:
        _ = execution_engine.run(
            runtime_parameters={
                "image_1": [license_plate_image, crowd_image],
                "reference": "people",
            }
        )

    # then
    assert (
        "roboflow_core/cosine_similarity@v1 block feed with different shape of embeddings"
        in str(error.value)
    )


CLIP_TEXT_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "prompt"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip@v1",
            "name": "embedding",
            "data": "$inputs.prompt",
            "version": "RN50",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "text_embeddings",
            "coordinates_system": "own",
            "selector": "$steps.embedding.embedding",
        },
    ],
}


def test_clip_text_embedding_model(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_TEXT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(runtime_parameters={"prompt": "Foo Bar"})

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 1, "Expected 1 element in the output"
    assert set(result[0].keys()) == {
        "text_embeddings",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["text_embeddings"]) == 1024
    ), "Expected text embedding to be of dimension 1024 for RN50 model"
    assert (
        pytest.approx(np.mean(result[0]["text_embeddings"]), 0.0001) == -0.016772
    ), "Expected embedding to have a value similar to during testing"
    assert (
        pytest.approx(np.max(result[0]["text_embeddings"]), 0.0001) == 1.65736556
    ), "Expected embedding to have a value similar to during testing"
    assert (
        pytest.approx(np.min(result[0]["text_embeddings"]), 0.0001) == -10.109556
    ), "Expected embedding to have a value similar to during testing"
    assert (
        pytest.approx(np.std(result[0]["text_embeddings"]), 0.0001) == 0.39733439
    ), "Expected embedding to have a value similar to during testing"


CLIP_COMPARISON_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
    ],
    "steps": [
        {
            "type": "ClipComparison",
            "name": "comparison",
            "images": "$inputs.image",
            "texts": "$inputs.reference",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "similarity",
            "selector": "$steps.comparison.similarity",
        },
    ],
}


@add_to_workflows_gallery(
    category="Basic Workflows",
    use_case_title="Workflow with CLIP Comparison",
    use_case_description="""
This is the basic workflow that only contains a single CLIP Comparison block. 

Please take a look at how batch-oriented WorkflowImage data is plugged to 
detection step via input selector (`$inputs.image`) and how non-batch parameters 
(reference set of texts that the each image in batch will be compared to)
is dynamically specified - via `$inputs.reference` selector.
    """,
    workflow_definition=CLIP_COMPARISON_WORKFLOW,
    workflow_name_in_app="clip",
)
def test_clip_workflow_when_minimal_valid_input_provided(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=CLIP_COMPARISON_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "reference": ["car", "crowd"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "similarity",
    }, "Expected all declared outputs to be delivered"
    assert (
        len(result[0]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for first image"
    assert (
        result[0]["similarity"][0] > result[0]["similarity"][1]
    ), "Expected to predict `car` class for first image"
    assert (
        len(result[1]["similarity"]) == 2
    ), "Expected 2 elements of similarity comparison list for second image"
    assert (
        result[1]["similarity"][0] < result[1]["similarity"][1]
    ), "Expected to predict `crowd` class for second image"


WORKFLOW_WITH_CLIP_COMPARISON_V2 = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {"type": "WorkflowParameter", "name": "reference"},
        {"type": "WorkflowParameter", "name": "version", "default_value": "ViT-B-16"},
    ],
    "steps": [
        {
            "type": "roboflow_core/clip_comparison@v2",
            "name": "comparison",
            "images": "$inputs.image",
            "classes": "$inputs.reference",
            "version": "$inputs.version",
        },
        {
            "type": "PropertyDefinition",
            "name": "property_extraction",
            "data": "$steps.comparison.classification_predictions",
            "operations": [
                {"type": "ClassificationPropertyExtract", "property_name": "top_class"}
            ],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "clip_output",
            "selector": "$steps.comparison.*",
        },
        {
            "type": "JsonField",
            "name": "class_name",
            "selector": "$steps.property_extraction.output",
        },
    ],
}


def test_workflow_with_clip_comparison_v2_and_property_definition_with_valid_input(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [license_plate_image, crowd_image],
            "reference": ["car", "crowd"],
        }
    )

    # then
    assert isinstance(result, list), "Expected list to be delivered"
    assert len(result) == 2, "Expected 2 elements in the output for two input images"
    assert set(result[0].keys()) == {
        "clip_output",
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert set(result[1].keys()) == {
        "clip_output",
        "class_name",
    }, "Expected all declared outputs to be delivered"
    assert np.allclose(
        result[0]["clip_output"]["similarities"],
        [0.23334351181983948, 0.17259158194065094],
        atol=1e-4,
    ), "Expected predicted similarities to match values verified at test creation"
    assert (
        abs(
            result[0]["clip_output"]["similarities"][0]
            - result[0]["clip_output"]["max_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        abs(
            result[0]["clip_output"]["similarities"][1]
            - result[0]["clip_output"]["min_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        result[0]["clip_output"]["most_similar_class"] == "car"
    ), "Expected most similar class to be extracted properly"
    assert (
        result[0]["clip_output"]["least_similar_class"] == "crowd"
    ), "Expected least similar class to be extracted properly"
    assert (
        result[0]["clip_output"]["classification_predictions"]["top"] == "car"
    ), "Expected classifier output to be shaped correctly"
    assert (
        result[0]["class_name"] == "car"
    ), "Expected property definition step to cooperate nicely with clip output"
    assert np.allclose(
        result[1]["clip_output"]["similarities"],
        [0.18426208198070526, 0.207647442817688],
        atol=1e-4,
    ), "Expected predicted similarities to match values verified at test creation"
    assert (
        abs(
            result[1]["clip_output"]["similarities"][1]
            - result[1]["clip_output"]["max_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        abs(
            result[1]["clip_output"]["similarities"][0]
            - result[1]["clip_output"]["min_similarity"]
        )
        < 1e-5
    ), "Expected max similarity to be correct"
    assert (
        result[1]["clip_output"]["most_similar_class"] == "crowd"
    ), "Expected most similar class to be extracted properly"
    assert (
        result[1]["clip_output"]["least_similar_class"] == "car"
    ), "Expected least similar class to be extracted properly"
    assert (
        result[1]["clip_output"]["classification_predictions"]["top"] == "crowd"
    ), "Expected classifier output to be shaped correctly"
    assert (
        result[1]["class_name"] == "crowd"
    ), "Expected property definition step to cooperate nicely with clip output"


def test_workflow_with_clip_comparison_v2_and_property_definition_with_empty_class_list(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": [license_plate_image, crowd_image],
                "reference": [],
            }
        )


def test_workflow_with_clip_comparison_v2_and_property_definition_with_invalid_model_version(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=WORKFLOW_WITH_CLIP_COMPARISON_V2,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    with pytest.raises(RuntimeInputError):
        _ = execution_engine.run(
            runtime_parameters={
                "image": [license_plate_image, crowd_image],
                "reference": ["car", "crowd"],
                "version": "invalid",
            }
        )
