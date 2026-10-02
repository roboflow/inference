from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from roboflow_workflows.execution_engine.core import ExecutionEngine


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_sliced_images_embed_and_connect_to_existing_cosine_similarity(output_type):
    manager = MagicMock()

    def infer(model_id, images, api_key=None, output_type="feature_vector"):
        assert output_type == expected_output_type
        return {
            "embeddings": [[2.0, 3.0] for image in images],
            "embedding_info": {"space_id": "same-space", "output_type": output_type},
        }

    expected_output_type = output_type
    manager.run_image_embeddings.side_effect = infer

    def infer_native(
        model_id,
        images,
        *,
        input_color_format,
        api_key=None,
        output_type="feature_vector"
    ):
        result = infer(model_id, images, api_key=api_key, output_type=output_type)
        result["embeddings"] = torch.tensor(result["embeddings"])

        return result

    manager.run_tensor_image_embeddings.side_effect = infer_native
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "InferenceImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/image_slicer@v1",
                "name": "slice",
                "image": "$inputs.image",
                "slice_width": 16,
                "slice_height": 16,
                "overlap_ratio_width": 0,
                "overlap_ratio_height": 0,
            },
            {
                "type": "roboflow_core/embedding_model@v1",
                "name": "embedding",
                "data": "$steps.slice.slices",
                "model_id": "my-project/1",
                "output_type": output_type,
            },
            {
                "type": "roboflow_core/cosine_similarity@v1",
                "name": "compare",
                "embedding_1": "$steps.embedding.embedding",
                "embedding_2": "$steps.embedding.embedding",
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "embeddings",
                "selector": "$steps.embedding.embedding",
            },
            {
                "type": "JsonField",
                "name": "similarities",
                "selector": "$steps.compare.similarity",
            },
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": manager,
            "workflows_core.api_key": "key",
        },
    )
    results = engine.run(
        serialize_results=True,
        runtime_parameters={
            "image": [
                np.zeros((32, 32, 3), dtype=np.uint8),
                np.zeros((32, 32, 3), dtype=np.uint8),
            ]
        },
    )
    assert len(results) == 2
    for result in results:
        assert result["embeddings"] == [[2.0, 3.0]] * 4
        np.testing.assert_allclose(result["similarities"], [1.0] * 4)
