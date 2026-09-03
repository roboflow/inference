import base64
import io

import numpy as np
import pycocotools.mask as mask_utils
import pytest
import supervision as sv
from PIL import Image
from pydantic import ValidationError

from inference.core.workflows.core_steps.models.roboflow.semantic_segmentation.v2 import (
    BlockManifest,
    RoboflowSemanticSegmentationModelBlockV2,
)


@pytest.mark.parametrize("images_field_alias", ["images", "image"])
def test_semantic_segmentation_model_validation_when_minimalistic_config_is_provided(
    images_field_alias: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        images_field_alias: "$inputs.image",
        "model_id": "some/1",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result == BlockManifest(
        type="roboflow_core/roboflow_semantic_segmentation_model@v2",
        name="some",
        images="$inputs.image",
        model_id="some/1",
    )


@pytest.mark.parametrize("field", ["type", "name", "images", "model_id"])
def test_semantic_segmentation_model_validation_when_required_field_is_not_given(
    field: str,
) -> None:
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }
    del data[field]

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_invalid_type_provided() -> None:
    # given
    data = {
        "type": "invalid",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_model_id_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        "images": "$inputs.image",
        "model_id": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_images_selector_has_invalid_type() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        "images": "some",
        "model_id": "some/1",
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_when_custom_mode_missing_custom_confidence() -> (
    None
):
    # given
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "confidence_mode": "custom",
        "custom_confidence": None,
    }

    # when
    with pytest.raises(ValidationError):
        _ = BlockManifest.model_validate(data)


def test_semantic_segmentation_model_validation_accepts_best_confidence_mode() -> None:
    # model eval now produces thresholds for semantic segmentation, so "best"
    # is an accepted confidence mode (matching object detection / instance seg).
    data = {
        "type": "roboflow_core/roboflow_semantic_segmentation_model@v2",
        "name": "some",
        "images": "$inputs.image",
        "model_id": "some/1",
        "confidence_mode": "best",
    }

    # when
    result = BlockManifest.model_validate(data)

    # then
    assert result.confidence_mode == "best"


# --- _convert_to_sv_detections tests ---


def _encode_mask_as_base64_png(mask_array: np.ndarray) -> str:
    pil_img = Image.fromarray(mask_array.astype(np.uint8), mode="L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_convert_to_sv_detections_produces_per_class_detections() -> None:
    seg = np.zeros((100, 100), dtype=np.uint8)
    seg[10:40, 10:40] = 1
    seg[60:90, 60:90] = 2

    result = RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"1": "cat", "2": "dog"},
        }
    )

    assert isinstance(result, sv.Detections)
    assert len(result) == 2
    assert set(result.class_id.tolist()) == {1, 2}
    # masks are RLE-encoded, not stored as binary arrays
    assert result.mask is None
    assert "rle_mask" in result.data
    rle_masks = result.data["rle_mask"]
    assert len(rle_masks) == 2
    # decoded RLE masks must cover the original class pixels
    decoded = np.array([mask_utils.decode(rle).astype(bool) for rle in rle_masks])
    assert decoded.shape == (2, 100, 100)
    assert set(result.data["class_name"].tolist()) == {"cat", "dog"}
    # no confidence mask → defaults to 1.0
    assert result.confidence is not None
    assert (result.confidence == 1.0).all()


def test_convert_to_sv_detections_derives_confidence_from_mask() -> None:
    seg = np.zeros((50, 50), dtype=np.uint8)
    seg[10:40, 10:40] = 1
    conf = np.full((50, 50), 200, dtype=np.uint8)  # 200/255 ≈ 0.784

    result = RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "confidence_mask": _encode_mask_as_base64_png(conf),
            "class_map": {"1": "cat"},
        }
    )

    assert "confidence_mask" in result.data
    # One entry per detection so boolean filtering keeps working; each entry
    # is the shared full-frame confidence map.
    assert result.data["confidence_mask"].shape == (len(result),)
    assert result.data["confidence_mask"][0].shape == (50, 50)
    assert result.confidence is not None
    assert abs(float(result.confidence[0]) - 200 / 255.0) < 0.01


def test_convert_to_sv_detections_empty_when_all_background() -> None:
    seg = np.zeros((50, 50), dtype=np.uint8)

    result = RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections(
        {"segmentation_mask": _encode_mask_as_base64_png(seg), "class_map": {}}
    )

    assert len(result) == 0


BLOCK_CLS = RoboflowSemanticSegmentationModelBlockV2


def _reference_convert_pre_optimization(seg, conf, class_map, block_cls):
    """The pre-optimization per-class algorithm, kept inline as the parity
    reference: C-order scan + per-class astype + asfortranarray copies."""
    import pycocotools.mask as ref_mask_utils

    unique_class_ids = [cid for cid in np.unique(seg).tolist() if cid != 0]
    xyxy, rles, confidences = [], [], []
    for class_id in unique_class_ids:
        binary_mask = seg == class_id
        rows = np.where(np.any(binary_mask, axis=1))[0]
        cols = np.where(np.any(binary_mask, axis=0))[0]
        xyxy.append([cols[0], rows[0], cols[-1], rows[-1]])
        rle = ref_mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
        rles.append(rle["counts"].decode("utf-8"))
        if conf is not None:
            confidences.append(float(conf[binary_mask].mean()) / 255.0)
        else:
            confidences.append(1.0)
    return unique_class_ids, xyxy, rles, confidences


def _random_label_map(h=97, w=131, classes=(3, 7, 255), seed=42):
    rng = np.random.default_rng(seed)
    seg = np.zeros((h, w), dtype=np.uint8)
    for cid in classes:
        y0, x0 = rng.integers(0, h - 10), rng.integers(0, w - 10)
        seg[y0 : y0 + rng.integers(5, h // 2), x0 : x0 + rng.integers(5, w // 2)] = cid
    speckle = rng.random((h, w)) < 0.01
    seg[speckle] = rng.choice(classes)
    return seg


def test_convert_to_sv_detections_matches_pre_optimization_output() -> None:
    # the F-order-once + present_class_ids optimizations must be byte-identical
    # to the previous implementation: same RLE counts, xyxy, confidence
    seg = _random_label_map()
    rng = np.random.default_rng(7)
    conf = rng.integers(0, 256, size=seg.shape, dtype=np.uint8)

    exp_ids, exp_xyxy, exp_rles, exp_conf = _reference_convert_pre_optimization(
        seg, conf, {"3": "a", "7": "b"}, BLOCK_CLS
    )

    result = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "confidence_mask": _encode_mask_as_base64_png(conf),
            "class_map": {"3": "a", "7": "b"},
        }
    )

    assert result.class_id.tolist() == exp_ids
    assert result.xyxy.tolist() == [[float(v) for v in box] for box in exp_xyxy]
    assert [r["counts"] for r in result.data["rle_mask"]] == exp_rles
    # detections store confidence as float32; apply the same rounding to the
    # float64 reference values before demanding exact equality
    assert result.confidence.tolist() == np.array(exp_conf, dtype=np.float32).tolist()


def test_convert_to_sv_detections_with_present_class_ids_hint_matches_unhinted() -> (
    None
):
    seg = _random_label_map(classes=(1, 9))

    unhinted = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"1": "cat", "9": "dog"},
        }
    )
    hinted = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"1": "cat", "9": "dog"},
            "present_class_ids": np.unique(seg).tolist(),
        }
    )

    assert hinted.class_id.tolist() == unhinted.class_id.tolist()
    assert hinted.xyxy.tolist() == unhinted.xyxy.tolist()
    assert [r["counts"] for r in hinted.data["rle_mask"]] == [
        r["counts"] for r in unhinted.data["rle_mask"]
    ]


def test_convert_to_sv_detections_skips_stale_present_class_ids() -> None:
    seg = np.zeros((40, 40), dtype=np.uint8)
    seg[5:20, 5:20] = 2

    result = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"2": "cat"},
            # 5 promised but absent from the mask - must be skipped, not crash
            "present_class_ids": [0, 2, 5],
        }
    )

    assert result.class_id.tolist() == [2]


def test_convert_to_sv_detections_empty_when_hint_only_contains_background() -> None:
    seg = np.zeros((40, 40), dtype=np.uint8)

    result = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {},
            "present_class_ids": [0],
        }
    )

    assert len(result) == 0


def test_convert_to_sv_detections_ignores_empty_present_class_ids_hint() -> None:
    # an empty hint is never emitted for a real mask - fall back to scanning
    seg = np.zeros((40, 40), dtype=np.uint8)
    seg[5:20, 5:20] = 2

    result = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "class_map": {"2": "cat"},
            "present_class_ids": [],
        }
    )

    assert result.class_id.tolist() == [2]


def test_convert_to_sv_detections_numpy_masks_match_base64_path() -> None:
    # response_mask_format="numpy" must produce exactly what the PNG/base64
    # round-trip produces: same RLE counts, xyxy, class ids, confidence
    seg = _random_label_map(classes=(3, 7))
    rng = np.random.default_rng(13)
    conf = rng.integers(0, 256, size=seg.shape, dtype=np.uint8)
    class_map = {"3": "a", "7": "b"}

    via_b64 = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "confidence_mask": _encode_mask_as_base64_png(conf),
            "class_map": class_map,
        }
    )
    via_numpy = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": seg,
            "confidence_mask": conf,
            "class_map": class_map,
        }
    )

    assert via_numpy.class_id.tolist() == via_b64.class_id.tolist()
    assert via_numpy.xyxy.tolist() == via_b64.xyxy.tolist()
    assert via_numpy.confidence.tolist() == via_b64.confidence.tolist()
    assert [r["counts"] for r in via_numpy.data["rle_mask"]] == [
        r["counts"] for r in via_b64.data["rle_mask"]
    ]
    assert len(via_numpy.data["confidence_mask"]) == len(
        via_b64.data["confidence_mask"]
    )
    assert np.array_equal(
        via_numpy.data["confidence_mask"][0], via_b64.data["confidence_mask"][0]
    )


def test_convert_to_sv_detections_numpy_empty_mask_yields_empty_detections() -> None:
    result = BLOCK_CLS._convert_to_sv_detections(
        {
            "segmentation_mask": np.zeros((32, 32), dtype=np.uint8),
            "confidence_mask": np.full((32, 32), 128, dtype=np.uint8),
            "class_map": {},
        }
    )

    assert len(result) == 0


def test_convert_to_sv_detections_output_survives_boolean_filtering() -> None:
    # Regression: the confidence map used to be stored as a bare (H, W) array
    # in `data`, so filtering the detections (which indexes every data field
    # with a length-N boolean mask) failed with "boolean index did not match
    # indexed array along axis 0".
    seg = np.zeros((50, 50), dtype=np.uint8)
    seg[5:20, 5:20] = 1
    seg[30:45, 30:45] = 2
    conf = np.full((50, 50), 200, dtype=np.uint8)

    result = RoboflowSemanticSegmentationModelBlockV2._convert_to_sv_detections(
        {
            "segmentation_mask": _encode_mask_as_base64_png(seg),
            "confidence_mask": _encode_mask_as_base64_png(conf),
            "class_map": {"1": "cat", "2": "dog"},
        }
    )
    assert len(result) == 2

    filtered = result[np.array([False, True])]

    assert len(filtered) == 1
    assert filtered.data["class_name"].tolist() == ["dog"]
    assert filtered.data["confidence_mask"][0].shape == (50, 50)
    assert filtered.data["confidence_mask"][0] is result.data["confidence_mask"][1]
