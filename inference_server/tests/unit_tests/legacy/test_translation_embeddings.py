import hashlib
import json

import numpy as np
import pytest

from inference_models.errors import ModelInputError
from inference_server.legacy.entities import ImageEmbeddingRequest
from inference_server.legacy.translation import (
    IMAGE_EMBEDDING_TASK_TYPES,
    build_image_embedding_params,
    make_embedding_info,
    repack_image_embeddings,
    require_feature_vector_output,
    text_image_embeddings_enabled,
    text_image_embeddings_for,
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
GRAYSCALE_OFF = {**NO_OVERRIDES, "disable_preproc_grayscale": True}
PRETRAINED_PREPROCESSING = {
    "image_processor_type": "BitImageProcessor",
    "do_convert_rgb": True,
    "do_resize": True,
    "size": {"shortest_edge": 256},
    "resample": 3,
    "do_center_crop": True,
    "crop_size": {"height": 224, "width": 224},
    "do_rescale": True,
    "rescale_factor": 0.00392156862745098,
    "do_normalize": True,
    "image_mean": [0.485, 0.456, 0.406],
    "image_std": [0.229, 0.224, 0.225],
}
CLIP_PREPROCESSING = {
    "image_size": 224,
    "resize": "shortest_edge",
    "resample": "bicubic",
    "center_crop": True,
    "rescale_factor": 1 / 255,
    "image_mean": [0.48145466, 0.4578275, 0.40821073],
    "image_std": [0.26862954, 0.26130258, 0.27577711],
}
CLASSIFIER_PREPROCESSING = {
    "image_pre_processing": {
        "auto-orient": {"enabled": True},
        "grayscale": {"enabled": True},
    },
    "network_input": {"training_input_size": {"width": 224, "height": 224}},
}
# Computed once with legacy inference.core.models.embeddings.make_embedding_info;
# tests/inference/unit_tests/core/models/test_legacy_classifier_embeddings.py
# asserts the same ids for the legacy server.
GOLDEN_EMBEDDING_SPACES = [
    (
        "dinov2/small",
        "dinov2-cls-token@v1",
        384,
        PRETRAINED_PREPROCESSING,
        "18f23774f892940283119ce8640ee99fc1f94060e190c3edad304690824dd7d5",
    ),
    (
        "clip/ViT-B-16",
        "clip-image-embeds@v1",
        512,
        CLIP_PREPROCESSING,
        "1bc0eb86e1747bddf8f0b1c0b109365a8347a2b40bfbeaab302b122e7d726ac1",
    ),
    (
        "my-classifier/1",
        "classifier-linear-input@v1",
        2048,
        CLASSIFIER_PREPROCESSING,
        "b4e7f1b4acc1340e8bca7fcb58a13b61f404df9da5c8f30ef11645d78b1c84d0",
    ),
]


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


def test_image_embedding_task_types_cover_classifiers_and_pretrained_models():
    assert IMAGE_EMBEDDING_TASK_TYPES == {
        "classification",
        "multi-label-classification",
        "embedding",
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


@pytest.mark.parametrize(
    "model_id,feature_definition,dimension,preprocessing,space_id",
    GOLDEN_EMBEDDING_SPACES,
)
def test_embedding_space_ids_match_the_legacy_server(
    model_id, feature_definition, dimension, preprocessing, space_id
):
    info = {
        "feature_definition": feature_definition,
        "output_type": "feature_vector",
        "normalization": "none",
        "dimension": dimension,
        "preprocessing": preprocessing,
        "backend": "Backend",
        "precision": "torch.float32",
    }

    embedding_info = make_embedding_info(model_id, info, GRAYSCALE_OFF)

    assert embedding_info.space_id == space_id


@pytest.mark.parametrize(
    "preprocessing", [PRETRAINED_PREPROCESSING, CLIP_PREPROCESSING]
)
def test_make_embedding_info_keeps_fixed_model_preprocessing_as_is(preprocessing):
    info = {**INFO, "preprocessing": preprocessing}

    embedding_info = make_embedding_info("dinov2/small", info, GRAYSCALE_OFF)

    assert embedding_info.preprocessing == preprocessing
    assert (
        embedding_info.space_id
        == make_embedding_info("dinov2/small", info, NO_OVERRIDES).space_id
    )


def test_make_embedding_info_still_adds_overrides_to_classifier_preprocessing():
    info = make_embedding_info("ds/1", INFO, GRAYSCALE_OFF)

    assert info.preprocessing == {**PREPROCESSING, "overrides": GRAYSCALE_OFF}
    assert info.space_id == _space_id("ds/1", GRAYSCALE_OFF)


@pytest.mark.parametrize(
    "model_id,action",
    [
        ("clip/ViT-B-16", "embed_images_with_info"),
        ("perception_encoder/PE-Core-L14-336", None),
        ("perception-encoder/PE-Core-L14-336", None),
    ],
)
def test_text_image_models_keep_their_legacy_image_embedding_entry(model_id, action):
    text_image = text_image_embeddings_for(model_id)

    assert text_image is not None
    assert text_image.action == action


@pytest.mark.parametrize(
    "model_id", ["dinov2/small", "dinov3/vits16", "resnet50", "ds/1", "clip-ds/1"]
)
def test_other_models_embed_through_the_capability(model_id):
    assert text_image_embeddings_for(model_id) is None


@pytest.mark.parametrize(
    "model_id,flag",
    [
        ("clip/ViT-B-16", "CORE_MODELS_ENABLED"),
        ("clip/ViT-B-16", "CORE_MODEL_CLIP_ENABLED"),
        ("perception_encoder/PE-Core-L14-336", "CORE_MODEL_PE_ENABLED"),
    ],
)
def test_text_image_models_follow_the_legacy_core_model_flags(
    monkeypatch, model_id, flag
):
    from inference_server import configuration

    text_image = text_image_embeddings_for(model_id)
    assert text_image_embeddings_enabled(text_image) is True

    monkeypatch.setattr(configuration, flag, False)

    assert text_image_embeddings_enabled(text_image) is False


def test_require_feature_vector_output_rejects_logits_with_the_legacy_message():
    require_feature_vector_output("feature_vector")

    with pytest.raises(ModelInputError) as error:
        require_feature_vector_output("logits")

    assert error.value.args[0] == (
        "Pretrained embedding models return only `feature_vector` embeddings, "
        "requested `logits`. Logits are available for ResNet, ViT and DINOv3 "
        "classifiers."
    )
    assert error.value.help_url == (
        "https://inference-models.roboflow.com/errors/input-validation/"
        "#modelinputerror"
    )
