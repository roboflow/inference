import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.math.cosine_similarity.v1 import (
    BlockManifest,
    CosineSimilarityBlockV1,
)


def test_manifest_parsing_when_data_is_valid():
    # Given
    data = {
        "type": "roboflow_core/cosine_similarity@v1",
        "name": "cosine_step",
        "embedding_1": "$steps.clip_image.embedding",
        "embedding_2": "$steps.clip_text.embedding",
    }

    # When
    result = BlockManifest.model_validate(data)

    # Then
    assert result.type == "roboflow_core/cosine_similarity@v1"
    assert result.name == "cosine_step"
    assert result.embedding_1 == "$steps.clip_image.embedding"
    assert result.embedding_2 == "$steps.clip_text.embedding"


def test_manifest_parsing_when_data_is_invalid():
    # Given invalid data (not a valid embedding selector)
    data = {
        "type": "roboflow_core/cosine_similarity@v1",
        "name": "cosine_step",
        "embedding_1": "invalid_data",
        "embedding_2": "invalid_data",
    }

    # When / Then
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(data)


def test_cosine_similarity_block_run_identical_embeddings():
    # Given identical embeddings
    block = CosineSimilarityBlockV1()
    embedding_1 = [0.1, 0.3, 0.5]
    embedding_2 = [0.1, 0.3, 0.5]

    # When
    result = block.run(embedding_1=embedding_1, embedding_2=embedding_2)

    # Then
    # Cosine similarity should be close to 1.0 for identical vectors
    assert pytest.approx(result["similarity"], 0.0001) == 1.0


def test_cosine_similarity_block_run_orthogonal_embeddings():
    # Given orthogonal embeddings
    block = CosineSimilarityBlockV1()
    embedding_1 = [1.0, 0.0, 0.0]
    embedding_2 = [0.0, 1.0, 0.0]

    # When
    result = block.run(embedding_1=embedding_1, embedding_2=embedding_2)

    # Then
    # Cosine similarity should be close to 0.0 for orthogonal vectors
    assert pytest.approx(result["similarity"], 0.0001) == 0.0


def test_cosine_similarity_block_run_negative_correlation():
    # Given inversely correlated embeddings
    block = CosineSimilarityBlockV1()
    embedding_1 = [1.0, 1.0, 1.0]
    embedding_2 = [-1.0, -1.0, -1.0]

    # When
    result = block.run(embedding_1=embedding_1, embedding_2=embedding_2)

    # Then
    # Cosine similarity should be close to -1.0 for perfectly negatively correlated vectors
    assert pytest.approx(result["similarity"], 0.0001) == -1.0
