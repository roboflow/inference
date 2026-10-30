"""Execute an exported ONNX artifact through native model and request consumers."""

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from inference_models.models.rfdetr.rfdetr_oriented_object_detection_onnx import (
    RFDetrForOrientedObjectDetectionONNX,
    _corners,
)


def metadata():
    return {"classHeading": {"plane": True, "stone": False}, "classHeadingRevision": 3}


def inference_config(size=64):
    return {
        "image_pre_processing": {},
        "network_input": {
            "training_input_size": {"height": size, "width": size},
            "dynamic_spatial_size_supported": False,
            "color_mode": "rgb",
            "resize_mode": "letterbox",
            "padding_value": 114,
            "input_channels": 3,
            "scaling_factor": 255,
            "normalization": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        },
        "class_names_operations": [
            {"type": "class_name_removal", "class_name": "background_class83422"}
        ],
    }


def package(
    tmp_path, *, side=0, box=(0.5, 0.5, 0.4, 0.2, 0.3), malformed=None, class_index=1
):
    dets = np.array([[box]], np.float32)
    labels = np.array([[[100.0, -12.0, -12.0]]], np.float32)
    labels[0, 0, class_index] = 12.0
    heading = np.full((1, 1, 4), -10.0, np.float32)
    heading[0, 0, side] = 10.0
    values = {"dets": dets, "labels": labels, "heading": heading}
    if malformed:
        values[malformed][...] = np.nan
    nodes = [
        helper.make_node("Constant", [], [name], value=numpy_helper.from_array(v))
        for name, v in values.items()
    ]
    graph = helper.make_graph(
        nodes,
        "native-obb",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, [1, 3, 64, 64])],
        [
            helper.make_tensor_value_info(n, TensorProto.FLOAT, v.shape)
            for n, v in values.items()
        ],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    onnx.save(model, tmp_path / "weights.onnx")
    (tmp_path / "class_names.txt").write_text("background_class83422\nplane\nstone\n")
    (tmp_path / "inference_config.json").write_text(json.dumps(inference_config()))
    return RFDetrForOrientedObjectDetectionONNX.from_pretrained(
        str(tmp_path), oriented_box_metadata=metadata()
    )


@pytest.mark.parametrize("side", range(4))
@pytest.mark.parametrize("wh", [(0.4, 0.2), (0.2, 0.4), (0.3, 0.3)])
def test_all_front_sides_preserve_footprint_and_directed_squares(tmp_path, side, wh):
    model = package(tmp_path, side=side, box=(0.5, 0.5, *wh, 0.3))
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0.5)[0]
    assert result.class_id.tolist() == [
        0
    ]  # remapped class identity, never background output0
    assert result.class_names == ["plane"]
    assert result.heading.tolist() == [True]
    scale = (64 / 73 + 31 / 35) / 2
    center = [36.5, (32 - 16) / (31 / 35)]
    expected = torch.tensor([[*center, wh[0] * 64 / scale, wh[1] * 64 / scale, 0.3]])
    actual = _corners(result.obb)[0].numpy()
    reference = _corners(expected)[0].numpy()
    np.testing.assert_allclose(
        sorted(map(tuple, actual)), sorted(map(tuple, reference)), atol=1e-5
    )
    expected_angle = (0.3 + side * math.pi / 2 + math.pi) % (2 * math.pi) - math.pi
    assert result.obb[0, 4].item() == pytest.approx(expected_angle, abs=1e-6)
    assert result.obb[0, 2:4].tolist() == pytest.approx(
        [wh[side % 2] * 64 / scale, wh[1 - side % 2] * 64 / scale]
    )
    assert result.obb[0, :2].tolist() == pytest.approx([36.5, 16 / (31 / 35)])
    assert result.xyxy.dtype.is_floating_point


def test_detection_ids_are_unique_across_images_calls_and_models(tmp_path):
    model = package(tmp_path)
    image = np.zeros((35, 73, 3), np.uint8)
    results = [
        *model([image, image], confidence=0),
        *model(image, confidence=0),
        *package(tmp_path)(image, confidence=0),
    ]
    ids = [identity for result in results for identity in result.detection_id]
    assert len(ids) == 8
    assert all(ids)
    assert len(set(ids)) == len(ids)
    for result in results:
        assert (
            result.to_supervision().data["detection_id"].tolist() == result.detection_id
        )


def test_realized_preprocessing_matches_model_similarity_on_odd_image(tmp_path):
    model = package(tmp_path)
    image = np.zeros((35, 73, 3), np.uint8)
    image[17:19, 35:38] = 255
    arrays, metas = model.pre_process([image])
    meta = metas[0]
    assert meta.scale_width == 64 / 73
    assert meta.scale_height == 31 / 35
    assert meta.pad_top == 16
    assert arrays.shape == (1, 3, 64, 64)
    assert arrays.dtype == np.float32
    assert (
        model([image, image], confidence=0.5)[1].obb.tolist()
        == model(image, confidence=0.5)[0].obb.tolist()
    )


@pytest.mark.parametrize(
    "height,width,pixel_hash,transformed",
    [
        (
            35,
            73,
            "d25653931528abaaccae0c1623b7be03ef70baeac69d6b554a4e1a2f6aaac2f6",
            [27.51999855, 27.47000122, 14.79557228, 5.2432189],
        ),
        (
            73,
            35,
            "e4b2fbf4e9dc0f229cdfd0d208c65357b0dc225bf39e921bfabda636ceecc1c7",
            [29.32999992, 23.68000031, 7.09376717, 10.93585682],
        ),
        (
            79,
            101,
            "f112d0893712cccaa52a14568df33bc73268a0b0badaf72e4013ab3fbfe812f3",
            [27.52000046, 25.5, 14.71126556, 8.50504971],
        ),
        (
            1,
            99,
            "472d724451529a66f3bef12c7ce213b617e6cb1529dfa4559368a230011cdc1b",
            [27.52000046, 31.37000084, 18.74500084, 0.1399495],
        ),
        (
            64,
            64,
            "2ffe74f47a7bb7350e913f6b9259080cbe3cee97b2d313d5e2fe2942108d98e9",
            [27.52000046, 23.68000031, 14.72000027, 10.88000011],
        ),
    ],
)
def test_matches_pinned_producer_pixels_and_inverts_target_geometry(
    tmp_path, height, width, pixel_hash, transformed
):
    # Fixtures from RFDT c54293a Letterbox(64), applied to this gradient and
    # obb=[width*.43,height*.37,width*.23,height*.17,.3].
    image = (
        np.arange(height * width * 3, dtype=np.uint32).reshape(height, width, 3) % 251
    ).astype(np.uint8)
    model = package(tmp_path, box=(*[value / 64 for value in transformed], 0.3))
    inputs, _ = model.pre_process([image])
    rgb = np.rint(
        (
            inputs[0].transpose(1, 2, 0) * np.array([0.229, 0.224, 0.225])
            + np.array([0.485, 0.456, 0.406])
        )
        * 255
    ).astype(np.uint8)
    assert hashlib.sha256(rgb.tobytes()).hexdigest() == pixel_hash
    result = model(image)[0]
    assert result.obb[0].tolist() == pytest.approx(
        [width * 0.43, height * 0.37, width * 0.23, height * 0.17, 0.3], abs=1e-5
    )


@pytest.mark.parametrize("name", ["dets", "labels", "heading"])
def test_nonfinite_predictions_are_rejected(tmp_path, name):
    model = package(tmp_path, malformed=name)
    with pytest.raises(ValueError, match="finite"):
        model(np.zeros((64, 64, 3), np.uint8))


@pytest.mark.parametrize("field", ["classHeading", "classHeadingRevision"])
def test_missing_frozen_core_metadata_rejects(tmp_path, field):
    package(tmp_path)
    value = metadata()
    del value[field]
    with pytest.raises(ValueError):
        RFDetrForOrientedObjectDetectionONNX.from_pretrained(
            str(tmp_path), oriented_box_metadata=value
        )


def test_no_heading_policy_is_inferred_from_four_way_head(tmp_path):
    package(tmp_path)
    with pytest.raises(ValueError, match="orientedBoxMetadata"):
        RFDetrForOrientedObjectDetectionONNX.from_pretrained(str(tmp_path))


def test_axial_class_ignores_heading_head_and_keeps_ordered_geometry(tmp_path):
    model = package(tmp_path, side=1, class_index=2)
    result = model(np.zeros((64, 64, 3), np.uint8))[0]
    assert result.class_id.tolist() == [1]
    assert result.class_names == ["stone"]
    assert result.heading.tolist() == [False]
    assert result.obb[0].tolist() == pytest.approx([32, 32, 25.6, 12.8, 0.3])


def test_empty_confidence_result_keeps_geometry_rank(tmp_path):
    result = package(tmp_path)(np.zeros((64, 64, 3), np.uint8), confidence=1.0)[0]
    assert result.obb.shape == (0, 5)
    assert result.to_supervision().data["xyxyxyxy"].shape == (0, 4, 2)


def test_optional_actual_exported_model(tmp_path):
    fixture = os.environ.get("OBB_EXPORTED_ONNX")
    if fixture is None:
        pytest.skip("Set OBB_EXPORTED_ONNX to the actual RF-DETR export fixture")
    (tmp_path / "weights.onnx").symlink_to(Path(fixture))
    (tmp_path / "class_names.txt").write_text("background_class83422\nplane\nstone\n")
    (tmp_path / "inference_config.json").write_text(json.dumps(inference_config()))
    model = RFDetrForOrientedObjectDetectionONNX.from_pretrained(
        str(tmp_path), oriented_box_metadata=metadata()
    )
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0, max_detections=4)[0]
    assert result.obb.shape == (4, 5)
    assert torch.isfinite(result.obb).all()
    assert set(result.class_id.tolist()) <= {0, 1}
    assert len(set(result.detection_id)) == 4


def test_auto_loader_and_request_consumer(tmp_path, monkeypatch):
    package(tmp_path, side=3)
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "oriented-object-detection",
                "backend_type": "onnx",
                "oriented_box_metadata": metadata(),
            }
        )
    )
    from inference_models import AutoModel

    loaded = AutoModel.from_pretrained(
        str(tmp_path), allow_direct_local_storage_loading=True
    )
    assert isinstance(loaded, RFDetrForOrientedObjectDetectionONNX)
    from inference.core.entities.requests.inference import (
        InferenceRequestImage,
        OrientedObjectDetectionInferenceRequest,
    )
    from inference.core.models import inference_models_adapters as adapters

    monkeypatch.setattr(
        adapters, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
    )
    adapter = adapters.InferenceModelsOrientedObjectDetectionAdapter(str(tmp_path))
    request = OrientedObjectDetectionInferenceRequest(
        model_id="native/1",
        image=InferenceRequestImage(
            type="numpy_object", value=np.zeros((35, 73, 3), np.uint8)
        ),
        confidence=0.5,
        disable_active_learning=True,
    )
    response = adapter.infer_from_request(request)
    assert response.task_type == "oriented-object-detection"
    assert response.image.width == 73
    assert response.image.height == 35
    row = response.predictions[0]
    assert row.class_id == 0 and row.heading
    assert row.obb[:2] == pytest.approx([36.5, 16 / (31 / 35)])
    assert row.obb[4] == pytest.approx(0.3 - math.pi / 2, abs=1e-6)
    assert "x" not in row.model_dump()


def test_static_crop_preprocessing_is_inverted_before_response(tmp_path):
    package(tmp_path)
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {
                **inference_config(),
                "image_pre_processing": {
                    "static-crop": {
                        "enabled": True,
                        "x_min": 10,
                        "x_max": 60,
                        "y_min": 10,
                        "y_max": 30,
                    }
                },
            }
        )
    )
    model = RFDetrForOrientedObjectDetectionONNX.from_pretrained(
        str(tmp_path), oriented_box_metadata=metadata()
    )
    result = model(np.zeros((35, 73, 3), np.uint8), confidence=0.5)[0]
    assert result.obb[0, :2].tolist() == pytest.approx([25, 6.5])


@pytest.mark.parametrize("tensor", [False, True])
def test_workflow_block_executes_installed_onnx_request_and_native_transport(
    tmp_path, monkeypatch, tensor
):
    package(tmp_path, side=2)
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_architecture": "rfdetr",
                "task_type": "oriented-object-detection",
                "backend_type": "onnx",
                "oriented_box_metadata": metadata(),
            }
        )
    )
    from roboflow_workflows.core_steps.common.entities import StepExecutionMode
    from roboflow_workflows.core_steps.common.oriented import serialise_native_obb
    from roboflow_workflows.core_steps.models.roboflow.oriented_object_detection import (
        v1,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        Batch,
        ImageParentMetadata,
        OriginCoordinatesSystem,
        WorkflowImageData,
    )

    from inference.core.interfaces.workflows_models_provider import (
        ModelManagerModelsProvider,
    )
    from inference.core.models import inference_models_adapters as adapters

    monkeypatch.setattr(
        adapters, "ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES", True
    )
    monkeypatch.setattr(v1, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor)
    adapter = adapters.InferenceModelsOrientedObjectDetectionAdapter(str(tmp_path))

    class LocalManager:
        def add_model(self, **kwargs):
            pass

        def infer_from_request_sync(self, model_id, request, **kwargs):
            response = adapter.infer_from_request(request)
            self.detection_id = response[0].predictions[0].detection_id
            return response

    manager = LocalManager()
    block = v1.RoboflowOrientedObjectDetectionModelBlockV1(
        ModelManagerModelsProvider(manager), None, StepExecutionMode.LOCAL
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="image0"),
        workflow_root_ancestor_metadata=ImageParentMetadata(
            "root", OriginCoordinatesSystem(13, 17, 300, 200)
        ),
        numpy_image=np.zeros((35, 73, 3), np.uint8),
    )
    result = block.run(
        Batch([image], indices=[(0,)]),
        model_id="native/1",
        confidence=0.5,
        max_detections=3,
        class_filter=["plane"],
    )
    native = serialise_native_obb(result[0]["predictions"])
    assert native["task_type"] == "oriented-object-detection"
    assert native["predictions"][0]["obb"] == pytest.approx(
        [
            36.5,
            16 / (31 / 35),
            0.4 * 64 / ((64 / 73 + 31 / 35) / 2),
            0.2 * 64 / ((64 / 73 + 31 / 35) / 2),
            0.3 - math.pi,
        ],
        abs=1e-5,
    )
    assert native["predictions"][0]["detection_id"] == manager.detection_id
    assert native["predictions"][0]["parent_id"] == "image0"
    assert native["predictions"][0]["heading"]

    from roboflow_workflows.core_steps.common.tensor_native import (
        native_detections_to_root_coordinates,
    )
    from roboflow_workflows.core_steps.common.utils import (
        sv_detections_to_root_coordinates,
    )
    from roboflow_workflows.core_steps.transformations.per_class_confidence_filter import (
        v1 as confidence_numpy,
    )
    from roboflow_workflows.core_steps.transformations.per_class_confidence_filter import (
        v1_tensor as confidence_tensor,
    )

    filtering = confidence_tensor if tensor else confidence_numpy
    empty = filtering.PerClassConfidenceFilterBlockV1().run(
        predictions=[result[0]["predictions"]], class_thresholds={"plane": 1.0}
    )[0]["predictions"]
    rooted = (
        native_detections_to_root_coordinates(empty)
        if tensor
        else sv_detections_to_root_coordinates(empty)
    )
    serialized = serialise_native_obb(rooted)
    assert serialized["predictions"] == []
    assert serialized["image"] == {"width": 300, "height": 200}
    assert serialized["class_heading"] == native["class_heading"]


def test_unused_frozen_classes_are_retained_without_output_mapping(tmp_path):
    package(tmp_path)
    frozen = metadata()
    frozen["classHeading"]["unused"] = True
    loaded = RFDetrForOrientedObjectDetectionONNX.from_pretrained(
        str(tmp_path), oriented_box_metadata=frozen
    )
    result = loaded(np.zeros((64, 64, 3), np.uint8), confidence=0.5)[0]
    assert result.image_metadata["class_heading"]["unused"] is True


@pytest.mark.parametrize("task", ["oriented-object-detection", "obb-detection"])
def test_core_metadata_and_standard_train_package_survive_cache_and_offline(
    tmp_path, monkeypatch, requests_mock, task
):
    from inference_models.configuration import ROBOFLOW_API_HOST
    from inference_models.models.auto_loaders import (
        auto_negotiation,
        auto_resolution_cache,
        core,
        model_cache_paths,
    )
    from inference_models.weights_providers import offline_registry

    source = tmp_path / "source"
    source.mkdir()
    package(source, side=1)
    assert {p.name for p in source.iterdir()} == {
        "weights.onnx",
        "class_names.txt",
        "inference_config.json",
    }
    files = []
    for artifact in source.iterdir():
        content = artifact.read_bytes()
        url = f"https://obb-fixture.example/{artifact.name}"
        requests_mock.get(url, content=content)
        files.append(
            {
                "fileHandle": artifact.name,
                "downloadUrl": url,
                "md5Hash": hashlib.md5(content).hexdigest(),
            }
        )
    metadata_url = f"{ROBOFLOW_API_HOST}/models/v1/external/weights"
    requests_mock.get(
        metadata_url,
        json={
            "modelMetadata": {
                "type": "external-model-metadata-v1",
                "modelId": "native/1",
                "modelArchitecture": "rfdetr",
                "taskType": task,
                "orientedBoxMetadata": metadata(),
                "modelPackages": [
                    {
                        "type": "external-model-package-v1",
                        "packageId": "nativeonnx",
                        "packageManifest": {
                            "type": "onnx-model-package-v1",
                            "backendType": "onnx",
                            "quantization": "fp32",
                            "dynamicBatchSize": False,
                            "staticBatchSize": 1,
                            "opset": 17,
                        },
                        "packageFiles": files,
                        "trustedSource": True,
                    }
                ],
            }
        },
    )
    for module in (model_cache_paths, auto_resolution_cache, offline_registry):
        monkeypatch.setattr(module, "INFERENCE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(
        auto_negotiation,
        "get_selected_onnx_execution_providers",
        lambda: ["CPUExecutionProvider"],
    )
    monkeypatch.setattr(core, "OFFLINE_MODE_WARM_UP", True)
    options = dict(
        backend="onnx",
        device="cpu",
        api_key="fixture-key",
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    fresh = core.AutoModel.from_pretrained("native/1", **options)
    monkeypatch.setattr(core, "OFFLINE_MODE_WARM_UP", False)
    requests_mock.get(
        metadata_url, exc=AssertionError("metadata network request after warmup")
    )
    cached = core.AutoModel.from_pretrained("native/1", **options)
    monkeypatch.setattr(core, "OFFLINE_MODE", True)
    offline = core.AutoModel.from_pretrained(
        "native/1", use_auto_resolution_cache=False, **options
    )
    image = np.zeros((35, 73, 3), np.uint8)
    expected = fresh(image)[0]
    for loaded in (cached, offline):
        assert loaded.metadata == fresh.metadata
        result = loaded(image)[0]
        torch.testing.assert_close(result.obb, expected.obb)
        assert result.class_id.tolist() == [0]
        assert result.heading.tolist() == [True]

    from inference.core.entities.requests.inference import (
        OrientedObjectDetectionInferenceRequest,
    )
    from inference.core.models import inference_models_adapters as adapters
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    adapter = adapters.InferenceModelsOrientedObjectDetectionAdapter(
        "native/1",
        api_key="fixture-key",
        onnx_execution_providers=["CPUExecutionProvider"],
    )
    server_url = "http://native-sdk-fixture"
    requests_mock.get(
        f"{server_url}/model/registry",
        json={
            "models": [
                {
                    "model_id": "native/1",
                    "task_type": "oriented-object-detection",
                    "batch_size": 1,
                    "input_height": 64,
                    "input_width": 64,
                }
            ]
        },
    )

    def infer_request(request, context):
        payload = request.json()
        assert payload["confidence"] == 0.5
        payload["disable_active_learning"] = True
        parsed = OrientedObjectDetectionInferenceRequest.model_validate(payload)
        response = adapter.infer_from_request(parsed)
        return response.model_dump(by_alias=True, exclude_none=True)

    requests_mock.post(
        f"{server_url}/infer/oriented_object_detection", json=infer_request
    )
    client = InferenceHTTPClient(
        api_url=server_url, api_key="fixture-key"
    ).select_api_v1()
    client.configure(
        InferenceConfiguration(
            confidence_threshold=0.5, client_downsizing_disabled=False
        )
    )
    response = client.infer(image, model_id="native/1")
    assert response["task_type"] == "oriented-object-detection"
    assert response["image"] == {"width": 73, "height": 35}
    assert response["class_heading"] == metadata()["classHeading"]
    assert response["predictions"][0]["obb"] == pytest.approx(expected.obb[0].tolist())
    batch = client.infer([image, image], model_id="native/1")
    assert len(batch) == 2
    assert all(item["image"] == response["image"] for item in batch)
    assert all(
        item["predictions"][0]["obb"] == pytest.approx(expected.obb[0].tolist())
        for item in batch
    )
    if os.environ.get("OBB_SMOKE_RESPONSE"):
        Path(os.environ["OBB_SMOKE_RESPONSE"]).write_text(json.dumps(response))

    manifest = (
        Path(
            model_cache_paths.generate_model_package_cache_path(
                model_id="native/1", package_id="nativeonnx"
            )
        )
        / "model_config.json"
    )
    contents = json.loads(manifest.read_text())
    contents["oriented_box_metadata"]["classHeading"]["plane"] = False
    manifest.write_text(json.dumps(contents))
    from inference_models.errors import ModelPackageAlternativesExhaustedError

    with pytest.raises(
        ModelPackageAlternativesExhaustedError, match="Offline oriented-box semantics"
    ):
        core.AutoModel.from_pretrained(
            "native/1", use_auto_resolution_cache=False, **options
        )


@pytest.mark.parametrize("task", ["oriented-object-detection", "object-detection"])
def test_request_factory_preserves_native_or_ordinary_request(task):
    from inference.core.entities.requests.inference import (
        ObjectDetectionInferenceRequest,
        OrientedObjectDetectionInferenceRequest,
        request_from_type,
    )

    request_class = (
        OrientedObjectDetectionInferenceRequest
        if task == "oriented-object-detection"
        else ObjectDetectionInferenceRequest
    )
    original = request_class(
        model_id="native/1",
        image={"type": "base64", "value": "AA=="},
        confidence=0.5,
        class_filter=["plane"],
        id="dispatch-request",
    )
    restored = request_from_type(task, original.model_dump())
    assert type(restored) is request_class
    assert restored.model_dump() == original.model_dump()
