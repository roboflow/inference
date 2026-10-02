from threading import Lock
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from inference.core.entities.requests.embeddings import ImageEmbeddingRequest
from inference.models.resnet.resnet_classification import ResNetClassification


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
