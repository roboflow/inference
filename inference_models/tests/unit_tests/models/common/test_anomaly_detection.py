import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from inference_models.errors import CorruptedModelPackageError, ModelInputError
from inference_models.models.common.anomaly_detection import (
    AnomalyCalibration,
    AnomalyModelManifest,
    AnomalyRawPrediction,
    parse_anomaly_model_manifest,
    post_process_anomaly_scores,
    pre_process_anomaly_images,
)

IMAGE_SIZE = 32


def manifest(**overrides) -> dict:
    return {
        "schema_version": 1,
        "config": {
            "architecture": "patchcore",
            "image_size": IMAGE_SIZE,
            "batch_size": 8,
            "neighbors": 1,
            "feature_layer": 3,
            "top_k": 10,
        },
        "calibration": {"threshold": 2.0, "scale": 0.5},
        "class_names": ["normal", "anomalous"],
        "preprocessing": "rgb-stretch-bilinear-imagenet-v1",
        **overrides,
    }


def export_pre_processing(**overrides) -> dict:
    return {
        "kind": "roboflow-folder-export-v1",
        "source_resize": None,
        "online_resize": None,
        "online_reencode": False,
        **overrides,
    }


def normalize(rgb_image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(rgb_image.copy()).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def raw_prediction(scores: list, size_hw=(10, 20)) -> AnomalyRawPrediction:
    return AnomalyRawPrediction(
        scores=torch.tensor(scores),
        maps=torch.rand(len(scores), IMAGE_SIZE, IMAGE_SIZE),
        original_sizes_hw=[size_hw] * len(scores),
    )


def test_manifest_schema_version_must_match_pre_processing_contract(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "inference_config.json"
    config_path.write_text(json.dumps(manifest(schema_version=2)))

    with pytest.raises(CorruptedModelPackageError, match="schema version"):
        parse_anomaly_model_manifest(str(config_path), "patchcore")


def test_manifest_with_unknown_class_names_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "inference_config.json"
    config_path.write_text(json.dumps(manifest(class_names=["ok", "defect"])))

    with pytest.raises(CorruptedModelPackageError, match="class_names"):
        parse_anomaly_model_manifest(str(config_path), "patchcore")


def test_bgr_arrays_and_rgb_tensors_produce_the_same_network_input() -> None:
    parsed = AnomalyModelManifest.model_validate(manifest())
    rgb_image = np.random.default_rng(0).integers(0, 255, (20, 40, 3), dtype=np.uint8)
    bgr_image = np.ascontiguousarray(rgb_image[:, :, ::-1])
    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1)

    from_array = pre_process_anomaly_images(bgr_image, parsed, torch.device("cpu"))
    from_tensor = pre_process_anomaly_images([rgb_tensor], parsed, torch.device("cpu"))

    expected = normalize(
        np.asarray(
            Image.fromarray(rgb_image).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR
            )
        )
    )
    assert torch.equal(from_array.network_input[0], expected)
    assert torch.equal(from_tensor.network_input[0], expected)
    assert from_array.original_sizes_hw == [(20, 40)]


def test_schema_2_reproduces_platform_export_before_network_resize() -> None:
    parsed = AnomalyModelManifest.model_validate(
        manifest(
            schema_version=2,
            preprocessing=export_pre_processing(
                source_resize=[24, 16], online_resize=[12, 8], online_reencode=True
            ),
        )
    )
    rgb_image = np.random.default_rng(1).integers(0, 255, (40, 60, 3), dtype=np.uint8)

    result = pre_process_anomaly_images(
        rgb_image, parsed, torch.device("cpu"), input_color_format="rgb"
    )

    exported = cv2.resize(rgb_image, (24, 16), interpolation=cv2.INTER_AREA)
    buffer = BytesIO()
    Image.fromarray(exported).save(buffer, format="JPEG", quality=75)
    exported = np.asarray(Image.open(buffer).convert("RGB"))
    exported = cv2.resize(exported, (12, 8), interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(exported, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )
    exported = Image.open(BytesIO(encoded.tobytes())).convert("RGB")
    expected = normalize(
        np.asarray(exported.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR))
    )
    assert torch.equal(result.network_input[0], expected)
    assert result.original_sizes_hw == [(40, 60)]


def test_float_images_are_rejected() -> None:
    parsed = AnomalyModelManifest.model_validate(manifest())

    with pytest.raises(ModelInputError, match="uint8"):
        pre_process_anomaly_images(
            np.zeros((20, 40, 3), dtype=np.float32), parsed, torch.device("cpu")
        )


def test_score_at_threshold_is_anomalous_with_confidence_one_half() -> None:
    calibration = AnomalyCalibration(threshold=2.0, scale=0.5)

    prediction = post_process_anomaly_scores(
        raw_prediction([1.0, 2.0, 1e9]), calibration, include_anomaly_map=False
    )

    assert prediction.class_id.tolist() == [0, 1, 1]
    assert prediction.confidence[1].tolist() == [0.5, 0.5]
    assert prediction.confidence.argmax(dim=1).tolist()[0] == 0
    assert prediction.confidence[2].tolist() == [0.0, 1.0]
    assert [m["is_anomalous"] for m in prediction.images_metadata] == [
        False,
        True,
        True,
    ]
    assert prediction.images_metadata[0]["anomaly_score"] == 1.0
    assert prediction.images_metadata[0]["anomaly_threshold"] == 2.0
    assert "anomaly_map" not in prediction.images_metadata[0]


def test_anomaly_map_is_returned_in_original_image_coordinates() -> None:
    calibration = AnomalyCalibration(threshold=2.0, scale=0.5)

    prediction = post_process_anomaly_scores(
        raw_prediction([1.0], size_hw=(10, 20)), calibration, include_anomaly_map=True
    )

    assert prediction.images_metadata[0]["anomaly_map"].shape == (10, 20)
    assert prediction.images_metadata[0]["anomaly_map"].dtype == np.float32


def test_non_finite_scores_are_rejected() -> None:
    calibration = AnomalyCalibration(threshold=2.0, scale=0.5)

    with pytest.raises(CorruptedModelPackageError, match="non-finite"):
        post_process_anomaly_scores(
            raw_prediction([float("nan")]), calibration, include_anomaly_map=False
        )
