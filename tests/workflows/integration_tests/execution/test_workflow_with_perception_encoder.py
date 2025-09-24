import numpy as np
import pytest

from inference.core.env import WORKFLOWS_MAX_CONCURRENT_STEPS
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.core import ExecutionEngine

PERCEPTION_ENCODER_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "InferenceImage", "name": "image_1"},
        {"type": "InferenceImage", "name": "image_2"},
    ],
    "steps": [
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedding_1",
            "data": "$inputs.image_1",
            "version": "PE-Core-L14-336",
        },
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedding_2",
            "data": "$inputs.image_2",
            "version": "PE-Core-L14-336",
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
def test_perception_encoder_embedding_model(
    model_manager: ModelManager,
    license_plate_image: np.ndarray,
    crowd_image: np.ndarray,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PERCEPTION_ENCODER_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(
        runtime_parameters={"image_1": license_plate_image, "image_2": crowd_image}
    )

    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0].keys()) == {"similarity", "image_embeddings"}
    assert isinstance(result[0]["similarity"], float)
    assert -1.0 <= result[0]["similarity"] <= 1.0
    assert len(result[0]["image_embeddings"]) >= 1024

PERCEPTION_ENCODER_TEXT_WORKFLOW = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowParameter", "name": "prompt"},
    ],
    "steps": [
        {
            "type": "roboflow_core/perception_encoder@v1",
            "name": "embedding",
            "data": "$inputs.prompt",
            "version": "PE-Core-L14-336",
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

@pytest.mark.skip(reason="Known problem of race condition in execution engine")
def test_perception_encoder_text_embedding_model(
    model_manager: ModelManager,
) -> None:
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": None,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=PERCEPTION_ENCODER_TEXT_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    result = execution_engine.run(runtime_parameters={"prompt": "Foo Bar"})

    assert isinstance(result, list)
    assert len(result) == 1
    assert set(result[0].keys()) == {"text_embeddings"}
    assert len(result[0]["text_embeddings"]) >= 1024
