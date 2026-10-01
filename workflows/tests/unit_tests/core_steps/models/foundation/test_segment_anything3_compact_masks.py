"""Opt-in compact masks preserve the SAM3 prediction and rendering contracts."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.serializers import serialise_rle_sv_detections
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v3,
    v3_tensor,
)
from roboflow_workflows.core_steps.visualizations.common.annotators.polygon import (
    PolygonAnnotator,
)
from roboflow_workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


@pytest.fixture
def sample():
    masks = np.zeros((5, 192, 256), dtype=np.uint8)
    masks[0, :30, :40] = 1
    masks[0, 5:15, 5:15] = 0  # hole
    masks[1, -20:, -25:] = 1  # right/bottom edges
    masks[2, 60:80, 50:70] = 1
    masks[2, 90:110, 100:120] = 1  # disconnected components
    masks[3, 0, -1] = 1  # single pixel
    # Last mask is empty; RLE output retains empty instances.
    rles = [mask_utils.encode(np.asfortranarray(mask)) for mask in masks]
    for rle in rles:
        rle["counts"] = rle["counts"].decode("ascii")
    payload = {
        "prompt_results": [
            {
                "prompt_index": 0,
                "predictions": [
                    {"masks": rle, "confidence": 0.9, "format": "rle"} for rle in rles
                ],
            }
        ]
    }
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=np.full((192, 256, 3), 50, dtype=np.uint8),
    )
    return masks.astype(bool), payload, image


def _block():
    return v3.SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test",
        step_execution_mode=StepExecutionMode.REMOTE,
    )


def _predictions(block, payload):
    return block._convert_rle_json_response_to_sv_detections(
        resp_json=payload,
        class_names=["box"],
        confidence=0.3,
        image_height=192,
        image_width=256,
    )


@pytest.mark.parametrize("module", [v3, v3_tensor])
def test_manifest_default_and_opt_in(module):
    fields = dict(type="roboflow_core/sam3@v3", name="sam", images="$inputs.image")
    assert module.BlockManifest(**fields).use_compact_masks is False
    assert (
        module.BlockManifest(**fields, use_compact_masks=True).use_compact_masks is True
    )


def test_conversion_rendering_and_serialization_parity_without_densifying(
    sample, monkeypatch
):
    masks, payload, image = sample
    block = _block()
    predictions = _predictions(block, payload)
    dense = block._post_process_result([image], [deepcopy(predictions)], "rle")[0][
        "predictions"
    ]
    np.testing.assert_array_equal(dense.mask, masks)
    expected_json = serialise_rle_sv_detections(dense)
    expected_polygon = PolygonAnnotator().annotate(image.numpy_image.copy(), dense)
    renderer = MaskVisualizationBlockV1()
    render_args = dict(
        image=image,
        copy_image=True,
        color_palette="DEFAULT",
        palette_size=10,
        custom_colors=[],
        color_axis="CLASS",
        opacity=0.5,
    )
    expected_mask = renderer.run(predictions=dense, **render_args)["image"].numpy_image

    def forbidden(*args, **kwargs):
        raise AssertionError("Compact path must not materialise full-frame masks")

    with monkeypatch.context() as patch:
        patch.setattr(mask_utils, "decode", forbidden)
        patch.setattr(sv.CompactMask, "to_dense", forbidden)
        patch.setattr(sv.CompactMask, "__array__", forbidden)
        patch.setattr(sv.CompactMask, "__getitem__", forbidden)
        compact = block._post_process_result(
            [image], [deepcopy(predictions)], "rle", use_compact_masks=True
        )[0]["predictions"]
        assert isinstance(compact.mask, sv.CompactMask)
        assert serialise_rle_sv_detections(compact) == expected_json
        np.testing.assert_array_equal(
            PolygonAnnotator().annotate(image.numpy_image.copy(), compact),
            expected_polygon,
        )
        np.testing.assert_array_equal(
            renderer.run(predictions=compact, **render_args)["image"].numpy_image,
            expected_mask,
        )
    np.testing.assert_array_equal(np.asarray(compact.mask), masks)
    np.testing.assert_array_equal(compact.xyxy, dense.xyxy)
    np.testing.assert_array_equal(compact.confidence, dense.confidence)
    np.testing.assert_array_equal(compact.class_id, dense.class_id)
    assert isinstance(dense.mask, np.ndarray)


@pytest.mark.parametrize("mode", ["sdk", "proxy", "local"])
def test_run_routes_opt_in_without_changing_model_request(sample, monkeypatch, mode):
    masks, payload, image = sample
    block = _block()
    monkeypatch.setattr(v3, "SAM3_EXEC_MODE", "remote" if mode == "proxy" else "local")
    client = MagicMock()
    client.sam3_concept_segment.return_value = [payload]
    monkeypatch.setattr(v3, "InferenceHTTPClient", MagicMock(return_value=client))
    response = MagicMock()
    response.json.return_value = payload
    monkeypatch.setattr(v3.requests, "post", MagicMock(return_value=response))
    if mode == "local":
        block._step_execution_mode = StepExecutionMode.LOCAL
        block._model_manager.run_sam3_segmentation.return_value = [
            SimpleNamespace(
                prompt_results=[
                    SimpleNamespace(
                        prompt_index=0,
                        predictions=[
                            SimpleNamespace(
                                masks=p["masks"], confidence=p["confidence"]
                            )
                            for p in payload["prompt_results"][0]["predictions"]
                        ],
                    )
                ]
            )
        ]
    for enabled in [False, True, False]:
        result = block.run(
            images=[image],
            model_id="sam3/sam3_final",
            class_names=["box"],
            confidence=0.3,
            class_mapping={"box": "product"},
            use_compact_masks=enabled,
        )[0]["predictions"]
        assert isinstance(result.mask, sv.CompactMask if enabled else np.ndarray)
        np.testing.assert_array_equal(np.asarray(result.mask), masks)
        assert result.data["class_name"].tolist() == ["product"] * 5
    if mode == "sdk":
        assert all(
            "use_compact_masks" not in call.kwargs
            for call in client.sam3_concept_segment.call_args_list
        )


def test_empty_predictions_and_polygon_output_are_unchanged(sample):
    _, _, image = sample
    block = _block()
    result = block._post_process_result([image], [sv.Detections.empty()], "rle", True)
    assert len(result[0]["predictions"]) == 0
    masks = np.ones((1, 192, 256), dtype=bool)
    predictions = sv.Detections(xyxy=np.array([[0, 0, 256, 192]]), mask=masks)
    result = block._post_process_result([image], [predictions], "polygons", True)
    assert result[0]["predictions"].mask is masks
