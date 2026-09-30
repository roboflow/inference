"""CPU regression tests for native SAM3 mask packing and NMS RLE reuse."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pycocotools import mask as mask_utils
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    rle as rle_module,
)
from roboflow_workflows.core_steps.models.foundation.segment_anything3 import (
    v1_tensor,
    v3_tensor,
)


@pytest.mark.parametrize("apply_nms", [False, True])
def test_v3_preserves_thresholds_mapping_and_reuses_nms_rles(apply_nms, monkeypatch):
    a = np.zeros((9, 12), dtype=bool)
    a[1:4, 2:6] = True
    b = np.zeros_like(a)
    b[6:8, 8:11] = True
    manager = Mock()
    manager.run_tensor_native_inference.return_value = [
        [
            {
                "prompt_index": 0,
                "masks": [
                    mask_utils.encode(np.asfortranarray(m, dtype=np.uint8))
                    for m in [a, b]
                ],
                "scores": [0.9, 0.2],
            },
            {
                "prompt_index": 1,
                "masks": [
                    mask_utils.encode(np.asfortranarray(m, dtype=np.uint8))
                    for m in [a, b]
                ],
                "scores": [0.8, 0.85],
            },
        ]
    ]
    image = SimpleNamespace(
        is_tensor_materialised=lambda: False,
        numpy_image=np.zeros((9, 12, 3), dtype=np.uint8),
        _read_shape_without_materialization=lambda: (9, 12),
    )
    monkeypatch.setattr(rle_module, "_assemble_detections", lambda **kwargs: kwargs)
    encode = Mock(wraps=mask_utils.encode)
    monkeypatch.setattr(mask_utils, "encode", encode)
    block = v3_tensor.SegmentAnything3BlockV3(manager, None, StepExecutionMode.LOCAL)
    result = block.run_locally(
        [image],
        "sam3",
        ["a", "b"],
        {"a": "mapped"},
        0.5,
        None,
        apply_nms,
        0.5,
        "rle",
    )[0]["predictions"]
    assert result["class_ids"] == ([0, 1] if apply_nms else [0, 1, 1])
    assert result["class_names_map"] == {0: "mapped", 1: "b"}
    assert encode.call_count == 0
    assert manager.run_tensor_native_inference.call_args.kwargs["mask_format"] == "rle"
    expected = [a, b] if apply_nms else [a, a, b]
    np.testing.assert_array_equal(
        mask_utils.decode(result["mask"].to_coco_rle_masks()).transpose(2, 0, 1),
        expected,
    )


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("apply_nms", [True, False])
def test_native_rle_matches_dense_threshold_nms_packing(seed, apply_nms, monkeypatch):
    from roboflow_workflows.core_steps.models.foundation.segment_anything3.v2_tensor import (
        _collect_from_native_with_nms,
    )
    from roboflow_workflows.core_steps.models.foundation.segment_anything_common.prompts import (
        Sam3Prompt,
    )

    rng = np.random.default_rng(seed)
    masks = rng.random((6, 18, 21)) > 0.5
    masks[1] = masks[0]  # suppression and equal-score tie cases
    scores = rng.choice([0.3, 0.5, 0.9], size=6).tolist()
    dense_results = [
        {"prompt_index": i, "masks": masks, "scores": scores} for i in range(3)
    ]
    rles = [mask_utils.encode(np.asfortranarray(m, dtype=np.uint8)) for m in masks]
    if seed % 2:
        rles = [
            {"size": r["size"], "counts": r["counts"].decode("ascii")} for r in rles
        ]
    native_results = [
        {"prompt_index": i, "masks": rles, "scores": scores} for i in range(3)
    ]
    prompts = [
        Sam3Prompt(type="text", text=str(i), output_prob_thresh=t)
        for i, t in enumerate([None, 0.5, 0.8])
    ]
    names = ["a", None, "c"]
    mapping = {"a": "renamed", "foreground": "", "c": "renamed"}
    image = SimpleNamespace(_read_shape_without_materialization=lambda: (18, 21))
    monkeypatch.setattr(v1_tensor, "_assemble_detections", lambda **kwargs: kwargs)
    monkeypatch.setattr(rle_module, "_assemble_detections", lambda **kwargs: kwargs)
    items = _collect_from_native_with_nms(
        dense_results, names, prompts, 0.3, apply_nms, 0.4
    )
    items = [(m, score, cid, mapping.get(name, name)) for m, score, cid, name in items]
    expected = v1_tensor._build_instance_detections(items, image, "rle")
    # The native builder must neither encode nor decode masks.
    with monkeypatch.context() as patch:
        patch.setattr(mask_utils, "encode", Mock(side_effect=AssertionError("encode")))
        patch.setattr(mask_utils, "decode", Mock(side_effect=AssertionError("decode")))
        actual = rle_module.build_native_rle_detections(
            native_results, names, mapping, prompts, 0.3, apply_nms, 0.4, image
        )
    for key in ["xyxy", "confidences", "class_ids", "class_names_map"]:
        assert actual[key] == expected[key]
    assert actual["mask"].masks == expected["mask"].masks


def test_native_rle_empty_masks_and_missing_prompts(monkeypatch):
    from roboflow_workflows.core_steps.models.foundation.segment_anything_common.prompts import (
        Sam3Prompt,
    )

    monkeypatch.setattr(rle_module, "_assemble_detections", lambda **kwargs: kwargs)
    empty = mask_utils.encode(np.zeros((5, 6), dtype=np.uint8, order="F"))
    actual = rle_module.build_native_rle_detections(
        [{"prompt_index": 0, "masks": [empty], "scores": [0.9]}],
        ["a", "b"],
        None,
        [Sam3Prompt(type="text", text=n) for n in ["a", "b"]],
        0.5,
        True,
        0.5,
        SimpleNamespace(_read_shape_without_materialization=lambda: (5, 6)),
    )
    assert actual["mask"].masks == []
    assert actual["xyxy"] == []


def test_real_native_carrier_metadata_and_rendering_match_dense_builder():
    from roboflow_workflows.core_steps.models.foundation.segment_anything_common.prompts import (
        Sam3Prompt,
    )
    from roboflow_workflows.core_steps.visualizations.common.base_tensor import (
        to_supervision_for_annotation,
    )
    from roboflow_workflows.execution_engine.entities.base import (
        ImageParentMetadata,
        WorkflowImageData,
    )

    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="frame"),
        numpy_image=np.zeros((30, 40, 3), dtype=np.uint8),
    )
    mask = np.zeros((30, 40), dtype=bool)
    mask[4:17, 8:29] = True
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    expected = v1_tensor._build_instance_detections(
        [(mask, 0.9, 0, "product")], image, "rle"
    )
    actual = rle_module.build_native_rle_detections(
        [{"prompt_index": 0, "masks": [rle], "scores": [0.9]}],
        ["box"],
        {"box": "product"},
        [Sam3Prompt(type="text", text="box")],
        0.3,
        True,
        0.5,
        image,
    )
    view, reference = to_supervision_for_annotation(
        actual
    ), to_supervision_for_annotation(expected)
    np.testing.assert_array_equal(view.xyxy, reference.xyxy)
    np.testing.assert_array_equal(view.confidence, reference.confidence)
    np.testing.assert_array_equal(view.class_id, reference.class_id)
    np.testing.assert_array_equal(view.mask.crop(0), reference.mask.crop(0))
    np.testing.assert_array_equal(view.data["class_name"], reference.data["class_name"])
