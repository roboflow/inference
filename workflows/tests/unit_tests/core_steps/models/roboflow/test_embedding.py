from unittest.mock import MagicMock

import numpy as np
import pytest
from pydantic import ValidationError
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.roboflow.embedding.v1 import (
    BlockManifest,
    EmbeddingModelBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    WorkflowImageData,
)


def image():
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="parent"),
        numpy_image=np.zeros((16, 16, 3), dtype=np.uint8),
    )


def response():
    return {
        "embeddings": [[2.0, 3.0], [4.0, 5.0]],
        "embedding_info": {
            "model_id": "my-project/1",
            "feature_definition": "classifier-linear-input@v1",
            "output_type": "feature_vector",
            "normalization": "none",
            "dimension": 2,
            "space_id": "same-space",
            "backend": "onnx",
            "precision": "float32",
        },
    }


def test_manifest_matches_clip_image_ports():
    manifest = BlockManifest(
        type="roboflow_core/embedding_model@v1",
        name="embedding",
        data="$steps.crop.crops",
        model_id="resnet101",
    )
    assert manifest.data == "$steps.crop.crops"
    assert manifest.output_type == "feature_vector"
    assert manifest.get_parameters_accepting_batches() == ["data"]
    assert [output.name for output in manifest.describe_outputs()] == [
        "embedding",
        "embedding_info",
    ]
    schema = manifest.model_json_schema()
    assert "include_diagnostics" not in schema["properties"]
    assert schema["required_model_capabilities"] == ["image_embeddings"]
    assert schema["compatible_model_architectures"] == ["resnet", "vit", "dinov3_probe"]
    assert schema["properties"]["output_type"]["enum"] == ["feature_vector", "logits"]
    with pytest.raises(ValidationError):
        BlockManifest(**{**manifest.model_dump(), "output_type": "probabilities"})
    with pytest.raises(ValidationError):
        BlockManifest(
            type="roboflow_core/embedding_model@v1",
            name="embedding",
            data="some text",
            model_id="resnet101",
        )


def test_list_and_tensor_manifests_offer_single_and_multi_label_classifiers():
    from roboflow_workflows.core_steps.models.roboflow.embedding.v1_tensor import (
        BlockManifest as TensorBlockManifest,
    )

    for manifest in (BlockManifest, TensorBlockManifest):
        assert manifest.get_compatible_task_types() == [
            "classification",
            "multi-label-classification",
        ]


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
def test_local_batch_uses_embedding_capability_and_preserves_order(output_type):
    manager = MagicMock()
    manager.run_image_embeddings.return_value = response()
    block = EmbeddingModelBlockV1(manager, "key", StepExecutionMode.LOCAL)
    result = block.run(
        Batch(indices=None, content=[image(), image()]),
        "my-project/1",
        output_type,
    )
    assert [item["embedding"] for item in result] == [[2.0, 3.0], [4.0, 5.0]]
    assert result[0]["embedding_info"]["normalization"] == "none"
    info = result[0]["embedding_info"]
    assert info["space_id"] == response()["embedding_info"]["space_id"]
    assert set(info) == {
        "model_id",
        "feature_definition",
        "output_type",
        "dimension",
        "normalization",
        "space_id",
    }
    manager.add_model.assert_called_once_with(
        "my-project/1",
        "key",
        required_capabilities=["image_embeddings"],
        output_type=output_type,
    )
    arguments = manager.run_image_embeddings.call_args.kwargs
    assert arguments["model_id"] == "my-project/1"
    assert arguments["api_key"] == "key"
    assert arguments["output_type"] == output_type
    assert len(arguments["images"]) == 2


def test_remote_forwards_output_type_and_returns_compact_metadata(monkeypatch):
    from roboflow_workflows.core_steps.models.roboflow.embedding import v1

    client = MagicMock()
    client.get_image_embeddings.return_value = {
        "embeddings": [[2.0, 3.0]],
        "embedding_info": {
            "output_type": "logits",
            "space_id": "same-space",
            "preprocessing": {"resize": "stretch"},
            "backend": "onnx",
            "precision": "float32",
            "feature_tensor": "classifier/output",
            "source_artifact_sha256": "artifact-hash",
            "transform_version": 1,
        },
    }
    monkeypatch.setattr(v1, "InferenceHTTPClient", MagicMock(return_value=client))
    block = EmbeddingModelBlockV1(MagicMock(), "key", StepExecutionMode.REMOTE)
    result = block.run(
        Batch(indices=None, content=[image()]),
        "my-project/1",
        "logits",
    )
    assert client.get_image_embeddings.call_args.kwargs["output_type"] == "logits"
    assert result[0]["embedding"] == [2.0, 3.0]
    info = result[0]["embedding_info"]
    assert info["space_id"] == "same-space"
    assert info == {"output_type": "logits", "space_id": "same-space"}


def test_empty_batch_does_not_load_model():
    manager = MagicMock()
    block = EmbeddingModelBlockV1(manager, "key", StepExecutionMode.LOCAL)
    assert block.run(Batch(indices=None, content=[]), "my-project/1") == []
    manager.add_model.assert_not_called()


def test_incorrect_response_count_fails():
    manager = MagicMock()
    manager.run_image_embeddings.return_value = response()
    block = EmbeddingModelBlockV1(manager, "key", StepExecutionMode.LOCAL)
    with pytest.raises(ValueError, match="count"):
        block.run(Batch(indices=None, content=[image()]), "my-project/1")


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
@pytest.mark.parametrize(
    "execution_mode", [StepExecutionMode.LOCAL, StepExecutionMode.REMOTE]
)
def test_tensor_variant_matches_tensor_embedding_ports(
    monkeypatch, output_type, execution_mode
):
    import torch
    from roboflow_workflows.core_steps.models.roboflow.embedding import v1
    from roboflow_workflows.core_steps.models.roboflow.embedding.v1_tensor import (
        EmbeddingModelBlockV1 as TensorEmbeddingBlock,
    )
    from roboflow_workflows.execution_engine.entities.tensor_native_types import (
        TENSOR_NATIVE_EMBEDDING_KIND,
    )

    manager = MagicMock()
    manager.run_image_embeddings.return_value = response()
    client = MagicMock()
    client.get_image_embeddings.return_value = [
        {"embeddings": [embedding], "embedding_info": response()["embedding_info"]}
        for embedding in response()["embeddings"]
    ]
    monkeypatch.setattr(v1, "InferenceHTTPClient", MagicMock(return_value=client))
    block = TensorEmbeddingBlock(manager, "key", execution_mode)
    result = block.run(
        Batch(indices=None, content=[image(), image()]), "my-project/1", output_type
    )
    assert block.get_manifest().describe_outputs()[0].kind == [
        TENSOR_NATIVE_EMBEDDING_KIND
    ]
    assert [item["embedding"].tolist() for item in result] == [[2.0, 3.0], [4.0, 5.0]]
    assert all(item["embedding"].dtype == torch.float32 for item in result)
    assert all(item["embedding"].shape == (2,) for item in result)
    assert all(item["embedding_info"]["space_id"] == "same-space" for item in result)
