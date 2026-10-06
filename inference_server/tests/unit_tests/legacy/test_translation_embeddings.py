import hashlib
import json

import numpy as np

from inference_server.legacy.entities import ImageEmbeddingRequest
from inference_server.legacy.translation import (
    IMAGE_EMBEDDING_TASK_TYPES,
    build_image_embedding_params,
    make_embedding_info,
    repack_image_embeddings,
)

IMG = {"type": "base64", "value": "x"}
PREPROCESSING = {
    "image_pre_processing": {"auto-orient": {"enabled": True}},
    "network_input": {"training_input_size": {"width": 224}},
}
INFO = {
    "feature_definition": "classifier-linear-input@v1",
    "output_type": "feature_vector",
    "normalization": "none",
    "dimension": 2,
    "preprocessing": PREPROCESSING,
    "backend": "ResNetForClassificationTorch",
    "precision": "torch.float32",
}
NO_OVERRIDES = {
    "disable_preproc_auto_orient": False,
    "disable_preproc_contrast": False,
    "disable_preproc_grayscale": False,
    "disable_preproc_static_crop": False,
}


def _space_id(model_id, overrides):
    identity = {
        "model_id": model_id,
        "feature_definition": "classifier-linear-input@v1",
        "dimension": 2,
        "normalization": "none",
        "preprocessing": {**PREPROCESSING, "overrides": overrides},
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_image_embedding_task_types_cover_both_classification_tasks():
    assert IMAGE_EMBEDDING_TASK_TYPES == {
        "classification",
        "multi-label-classification",
    }


def test_build_image_embedding_params_carries_output_type_and_disabled_steps():
    request = ImageEmbeddingRequest(
        model_id="ds/1",
        image=IMG,
        output_type="logits",
        disable_preproc_static_crop=True,
    )

    assert build_image_embedding_params(request) == {
        "output_type": "logits",
        "disable_preproc_static_crop": True,
    }


def test_make_embedding_info_hashes_the_space_identity():
    info = make_embedding_info("ds/1", INFO, NO_OVERRIDES)

    expected = _space_id("ds/1", NO_OVERRIDES)
    assert info.space_id == expected
    assert info.model_dump(exclude_none=True) == {
        **INFO,
        "preprocessing": {**PREPROCESSING, "overrides": NO_OVERRIDES},
        "model_id": "ds/1",
        "space_id": expected,
    }
    assert list(info.preprocessing) == [
        "image_pre_processing",
        "network_input",
        "overrides",
    ]


def test_make_embedding_info_keeps_null_overrides():
    overrides = {**NO_OVERRIDES, "disable_preproc_grayscale": None}

    info = make_embedding_info("ds/1", INFO, overrides)

    assert info.preprocessing["overrides"]["disable_preproc_grayscale"] is None
    assert info.space_id == _space_id("ds/1", overrides)
    assert info.space_id != _space_id("ds/1", NO_OVERRIDES)


def test_make_embedding_info_keeps_onnx_extraction_details():
    info = make_embedding_info(
        "ds/1",
        {
            **INFO,
            "feature_tensor": "features",
            "source_artifact_sha256": "b" * 64,
            "transform_version": 1,
        },
        NO_OVERRIDES,
    )

    assert info.feature_tensor == "features"
    assert info.source_artifact_sha256 == "b" * 64
    assert info.transform_version == 1
    assert info.space_id == make_embedding_info("ds/1", INFO, NO_OVERRIDES).space_id


def test_make_embedding_info_changes_with_the_model_and_preprocessing():
    base = make_embedding_info("ds/1", INFO, NO_OVERRIDES)

    assert make_embedding_info("ds/2", INFO, NO_OVERRIDES).space_id != base.space_id
    assert (
        make_embedding_info(
            "ds/1", INFO, {**NO_OVERRIDES, "disable_preproc_grayscale": True}
        ).space_id
        != base.space_id
    )


def test_repack_image_embeddings_stacks_vectors_and_echoes_request_overrides():
    request = ImageEmbeddingRequest(
        model_id="ds/1",
        image=[IMG, IMG],
        disable_preproc_auto_orient=True,
        disable_preproc_contrast=None,
    )
    results = [
        {
            "embeddings": np.array([[1.0, 2.0]], dtype=np.float32),
            "embedding_info": INFO,
        },
        {"embeddings": np.array([3.0, 4.0], dtype=np.float32), "embedding_info": INFO},
    ]

    response = repack_image_embeddings(results, "ds/1", request)

    assert response.embeddings == [[1.0, 2.0], [3.0, 4.0]]
    assert response.inference_id == request.id
    assert response.embedding_info.model_id == "ds/1"
    assert response.embedding_info.preprocessing["overrides"] == {
        "disable_preproc_auto_orient": True,
        "disable_preproc_contrast": None,
        "disable_preproc_grayscale": False,
        "disable_preproc_static_crop": False,
    }
    assert response.time is None and response.resolved_model is None
