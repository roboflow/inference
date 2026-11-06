from threading import Lock
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.core.models.embeddings import make_embedding_info
from inference.models.resnet.resnet_classification import ResNetClassification

FEATURE_VECTOR = {"output_type": "feature_vector", "normalization": "none"}
# The same spaces, with the same golden ids, are asserted for the new server in
# inference_server/tests/unit_tests/legacy/test_translation_embeddings.py.
GOLDEN_EMBEDDING_SPACES = [
    (
        "dinov2/small",
        "dinov2-cls-token@v1",
        384,
        {
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
        },
        "18f23774f892940283119ce8640ee99fc1f94060e190c3edad304690824dd7d5",
    ),
    (
        "clip/ViT-B-16",
        "clip-image-embeds@v1",
        512,
        {
            "image_size": 224,
            "resize": "shortest_edge",
            "resample": "bicubic",
            "center_crop": True,
            "rescale_factor": 1 / 255,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
        },
        "1bc0eb86e1747bddf8f0b1c0b109365a8347a2b40bfbeaab302b122e7d726ac1",
    ),
    (
        "my-classifier/1",
        "classifier-linear-input@v1",
        2048,
        {
            "image_pre_processing": {
                "auto-orient": {"enabled": True},
                "grayscale": {"enabled": True},
            },
            "network_input": {"training_input_size": {"width": 224, "height": 224}},
            "overrides": {
                "disable_preproc_auto_orient": False,
                "disable_preproc_contrast": False,
                "disable_preproc_grayscale": True,
                "disable_preproc_static_crop": False,
            },
        },
        "b4e7f1b4acc1340e8bca7fcb58a13b61f404df9da5c8f30ef11645d78b1c84d0",
    ),
]


@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
@pytest.mark.parametrize("onnx_batch_size", [2, "batch"])
def test_legacy_embeddings_use_classifier_preprocessing_and_pad_static_batches(
    tmp_path,
    monkeypatch,
    output_type,
    onnx_batch_size,
):
    from inference.core.models import classification_base

    monkeypatch.setattr(classification_base, "MAX_BATCH_SIZE", 2)
    graph = helper.make_graph(
        [
            helper.make_node("GlobalAveragePool", ["image"], ["pooled"]),
            helper.make_node("Flatten", ["pooled"], ["features"]),
            helper.make_node("Gemm", ["features", "0.fc.weight"], ["logits"], transB=1),
            helper.make_node("Softmax", ["logits"], ["scores"], axis=1),
        ],
        "classifier",
        [
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, [onnx_batch_size, 3, 2, 2]
            )
        ],
        [
            helper.make_tensor_value_info(
                "scores", TensorProto.FLOAT, [onnx_batch_size, 2]
            )
        ],
        initializer=[
            numpy_helper.from_array(
                np.arange(6, dtype=np.float32).reshape(2, 3), "0.fc.weight"
            )
        ],
    )
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    source.ir_version = 9
    path = tmp_path / "best.onnx"
    onnx.save(source, path)
    before = path.read_bytes()
    model = ResNetClassification.__new__(ResNetClassification)
    model.endpoint = "project/1"
    model.preproc = {}
    model.cache_file = lambda filename: str(path)
    model._session_lock = Lock()
    model.onnx_session = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    model.input_name = "image"
    model.preproc_image = lambda image, **kwargs: (
        image["value"].astype(np.float32).transpose(2, 0, 1)[None],
        (2, 2),
    )
    values = (20, 50, 100, 150, 200)
    images = [np.full((2, 2, 3), value, dtype=np.uint8) for value in values]
    model.preprocess = MagicMock(wraps=model.preprocess)
    embedding_session, info = model.prepare_image_embeddings(output_type)
    wrapped_session = MagicMock(wraps=embedding_session)
    model._embedding_sessions[output_type] = (wrapped_session, info)
    request = ImageEmbeddingRequest(
        model_id="project/1",
        output_type=output_type,
        image=[{"type": "numpy_object", "value": image} for image in images],
    )
    response = model.infer_embeddings_from_request(request)
    expected = (
        np.array(values, dtype=np.float32)[:, None] / 255
        - np.array(model.preprocess_means)
    ) / np.array(model.preprocess_stds)
    if output_type == "logits":
        expected = expected @ np.arange(6, dtype=np.float32).reshape(2, 3).T
    np.testing.assert_allclose(response.embeddings, expected, atol=1e-6)
    assert response.embedding_info.model_id == "project/1"
    assert response.embedding_info.dimension == (
        3 if output_type == "feature_vector" else 2
    )
    assert response.embedding_info.output_type == output_type
    assert response.inference_id == request.id
    assert path.read_bytes() == before
    assert model.onnx_session.get_outputs()[0].name == "scores"
    assert [len(call.kwargs["image"]) for call in model.preprocess.call_args_list] == [
        2,
        2,
        1,
    ]
    assert [
        len(call.args[1]["image"]) for call in wrapped_session.run.call_args_list
    ] == ([2, 2, 2] if onnx_batch_size == 2 else [2, 2, 1])

    model.infer_embeddings_from_request = MagicMock(
        side_effect=AssertionError("HTTP conversion")
    )
    native = model.run_tensor_native_embeddings(
        images=images, input_color_format="bgr", output_type=output_type
    )
    np.testing.assert_allclose(native["embeddings"].numpy(), expected, atol=1e-6)
    assert native["embedding_info"] == response.embedding_info.model_dump(
        exclude_none=True
    )
    model.infer_embeddings_from_request.assert_not_called()


@pytest.mark.parametrize(
    "model_id,feature_definition,dimension,preprocessing,space_id",
    GOLDEN_EMBEDDING_SPACES,
)
def test_embedding_space_ids_match_the_new_server(
    model_id, feature_definition, dimension, preprocessing, space_id
):
    info = make_embedding_info(
        model_id=model_id,
        feature_info={"feature_definition": feature_definition, **FEATURE_VECTOR},
        preprocessing=preprocessing,
        backend="Backend",
        precision="torch.float32",
        dimension=dimension,
    )

    assert info.space_id == space_id
    assert info.preprocessing == preprocessing
