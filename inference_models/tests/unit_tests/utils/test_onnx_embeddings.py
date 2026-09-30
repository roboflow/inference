import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from inference_models.utils.onnx_embeddings import (
    extract_classifier_embedding,
    prepare_classifier_embedding,
)


def classifier(control_flow=False, matmul=False, activation="Softmax"):
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 3])]
    nodes = []
    if control_flow:
        inputs.append(helper.make_tensor_value_info("condition", TensorProto.BOOL, []))
        nodes.append(
            helper.make_node(
                "Constant",
                [],
                ["captured"],
                value=numpy_helper.from_array(np.ones(3, dtype=np.float32)),
            )
        )
        branch = helper.make_graph(
            [helper.make_node("Add", ["x", "captured"], ["branch_features"])],
            "branch",
            [],
            [
                helper.make_tensor_value_info(
                    "branch_features", TensorProto.FLOAT, ["batch", 3]
                )
            ],
        )
        nodes.append(
            helper.make_node(
                "If",
                ["condition"],
                ["features"],
                then_branch=branch,
                else_branch=branch,
            )
        )
    else:
        nodes.append(helper.make_node("Identity", ["x"], ["features"]))
    weight = np.arange(6, dtype=np.float32).reshape(2, 3)
    if matmul:
        weight = weight.T.copy()
        nodes.extend(
            [
                helper.make_node(
                    "MatMul", ["features", "classifier.weight"], ["product"]
                ),
                helper.make_node("Add", ["product", "classifier.bias"], ["logits"]),
            ]
        )
    else:
        nodes.append(
            helper.make_node(
                "Gemm",
                ["features", "classifier.weight", "classifier.bias"],
                ["logits"],
                transB=1,
            )
        )
    if activation:
        nodes.append(
            helper.make_node(
                activation,
                ["logits"],
                ["scores"],
                **({"axis": 1} if activation == "Softmax" else {}),
            )
        )
    graph = helper.make_graph(
        nodes,
        "classifier",
        inputs,
        [
            helper.make_tensor_value_info(
                "scores" if activation else "logits", TensorProto.FLOAT, ["batch", 2]
            )
        ],
        initializer=[
            numpy_helper.from_array(weight, "classifier.weight"),
            numpy_helper.from_array(
                np.array([-2, 1], dtype=np.float32), "classifier.bias"
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    return model


@pytest.mark.parametrize(
    "control_flow,matmul", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
@pytest.mark.parametrize("activation", ["Softmax", "Sigmoid", None])
def test_features_match_original_and_classifier_is_unchanged(
    control_flow, matmul, output_type, activation
):
    source = classifier(control_flow, matmul, activation)
    before = source.SerializeToString()
    extracted, info = extract_classifier_embedding(source, output_type)
    assert source.SerializeToString() == before
    reference = onnx.ModelProto()
    reference.CopyFrom(source)
    reference.graph.output.append(
        helper.make_tensor_value_info(
            "features" if output_type == "feature_vector" else "logits",
            TensorProto.FLOAT,
            ["batch", 3 if output_type == "feature_vector" else 2],
        )
    )
    image = np.arange(6, dtype=np.float32).reshape(2, 3)
    inputs = {"x": image}
    if control_flow:
        inputs["condition"] = np.array(True)
    reference_outputs = ort.InferenceSession(
        reference.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, inputs)
    embeddings = ort.InferenceSession(
        extracted.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, inputs)[0]
    np.testing.assert_array_equal(embeddings, reference_outputs[1])
    assert info["dimension"] == (3 if output_type == "feature_vector" else 2)
    assert info["output_type"] == output_type
    assert info["normalization"] == "none"
    assert not any(
        node.op_type in {"Softmax", "Sigmoid"} for node in extracted.graph.node
    )
    assert any(node.op_type in {"Gemm", "MatMul"} for node in extracted.graph.node) == (
        output_type == "logits"
    )


def test_unknown_head_is_rejected():
    source = classifier()
    source.graph.initializer[0].name = "unknown.weight"
    source.graph.node[1].input[1] = "unknown.weight"
    with pytest.raises(ValueError, match="weight tensor"):
        extract_classifier_embedding(source)


def test_shared_classifier_output_is_rejected():
    source = classifier()
    source.graph.node.append(helper.make_node("Identity", ["logits"], ["shared"]))
    with pytest.raises(ValueError, match="shared"):
        extract_classifier_embedding(source)


def test_artifact_cache_preserves_source_and_invalidates_for_changed_weights(tmp_path):
    path = tmp_path / "weights.onnx"
    source = classifier()
    onnx.save(source, path)
    before = path.read_bytes()
    first, first_info = prepare_classifier_embedding(str(path))
    assert prepare_classifier_embedding(str(path)) == (first, first_info)
    logits_path, logits_info = prepare_classifier_embedding(str(path), "logits")
    assert prepare_classifier_embedding(str(path), "logits") == (
        logits_path,
        logits_info,
    )
    assert logits_path != first
    assert logits_info["feature_definition"] != first_info["feature_definition"]
    assert path.read_bytes() == before
    source.graph.initializer[1].CopyFrom(
        numpy_helper.from_array(np.ones(2, dtype=np.float32), "classifier.bias")
    )
    onnx.save(source, path)
    second, second_info = prepare_classifier_embedding(str(path))
    assert first != second
    assert first_info["source_artifact_sha256"] != second_info["source_artifact_sha256"]


@pytest.mark.parametrize("embedding_only", [False, True])
@pytest.mark.parametrize("output_type", ["feature_vector", "logits"])
@pytest.mark.parametrize(
    "family,multi_label",
    [
        ("resnet", False),
        ("resnet", True),
        ("vit", False),
        ("vit", True),
        ("dinov3", False),
        ("dinov3", True),
    ],
)
def test_onnx_model_embedding_session_and_classification_both_work(
    tmp_path, embedding_only, family, multi_label, output_type
):
    import importlib

    import torch

    prefix = {"resnet": "ResNet", "vit": "VIT", "dinov3": "DinoV3"}[family]
    suffix = "MultiLabelClassificationOnnx" if multi_label else "ClassificationOnnx"
    model_class = getattr(
        importlib.import_module(
            f"inference_models.models.{family}.{family}_classification_onnx"
        ),
        f"{prefix}For{suffix}",
    )
    source_path = str(tmp_path / "weights.onnx")
    onnx.save(classifier(), source_path)
    (tmp_path / "class_names.txt").write_text("a\nb")
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                "image_pre_processing": {},
                "network_input": {
                    "training_input_size": {"height": 32, "width": 32},
                    "dynamic_spatial_size_supported": False,
                    "color_mode": "rgb",
                    "resize_mode": "stretch",
                    "input_channels": 3,
                    "scaling_factor": 255,
                    "normalization": [[0.5] * 3, [0.5] * 3],
                },
                "post_processing": {
                    "type": "sigmoid" if multi_label else "softmax",
                    "fused": True,
                },
            }
        )
    )
    providers = ["CPUExecutionProvider"]
    model = model_class.from_pretrained(
        str(tmp_path),
        onnx_execution_providers=providers,
        device=torch.device("cpu"),
        required_capabilities=["image_embeddings"] if embedding_only else None,
        output_type=output_type,
    )
    inputs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    features = model.forward_embedding(inputs)
    logits = inputs @ torch.arange(6, dtype=torch.float32).reshape(
        2, 3
    ).T + torch.tensor([-2, 1])
    torch.testing.assert_close(
        features, inputs if output_type == "feature_vector" else logits, rtol=0, atol=0
    )
    if embedding_only:
        assert model._session._session is None
    torch.testing.assert_close(
        model.forward_embedding(inputs, output_type="logits"), logits, rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.forward_embedding(inputs, output_type="feature_vector"),
        inputs,
        rtol=0,
        atol=0,
    )
    if embedding_only:
        assert model._session._session is None
    scores = model.forward(inputs)
    expected = torch.softmax(logits, dim=1)
    torch.testing.assert_close(scores, expected)
    assert model.embedding_info["dimension"] == (
        3 if output_type == "feature_vector" else 2
    )


def test_tensorrt_embedding_cache_is_separate_and_configuration_is_preserved():
    from inference_models.models.base.image_embeddings import embedding_providers

    providers = [
        (
            "TensorrtExecutionProvider",
            {
                "device_id": 2,
                "trt_engine_cache_path": "/classifier",
                "trt_fp16_enable": True,
            },
        ),
        "CUDAExecutionProvider",
    ]
    adapted = embedding_providers(providers, "/embedding/weights.onnx")
    assert adapted[0] == (
        "TensorrtExecutionProvider",
        {
            "device_id": 2,
            "trt_engine_cache_path": "/embedding",
            "trt_fp16_enable": True,
        },
    )
    assert providers[0][1]["trt_engine_cache_path"] == "/classifier"
    assert embedding_providers(["CPUExecutionProvider"], "/embedding/weights.onnx") == [
        ("CPUExecutionProvider", {})
    ]
