"""Real SAM3 postprocessor -> native workflow RLE parity (optional SAM3 extra)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize("apply_nms", [False, True])
@pytest.mark.parametrize("seed", range(4))
def test_real_postprocessor_rle_matches_dense_workflow(seed, apply_nms, monkeypatch):
    pytest.importorskip("sam3.eval.postprocessors")
    from inference.core.entities.requests.sam3 import Sam3Prompt
    from inference.core.workflows.core_steps.models.foundation.segment_anything3 import (
        rle,
        v1_tensor,
        v2_tensor,
    )
    from inference_models.models.sam3.chunked_postprocessing import (
        ChunkedPostProcessImage,
    )

    generator = torch.Generator().manual_seed(seed)
    outputs = {
        "pred_boxes": torch.rand(3, 12, 4, generator=generator) * 0.5 + 0.25,
        "pred_logits": torch.randn(3, 12, 1, generator=generator),
        "pred_masks": torch.randn(3, 12, 16, 16, generator=generator) * 4,
    }
    sizes = torch.tensor([[64, 48]] * 3)

    def postprocess(as_rle):
        post = ChunkedPostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=as_rle,
            detection_threshold=0.3,
            to_cpu=True,
            always_interpolate_masks_on_gpu=False,
            use_presence=False,
            mask_chunk_size=3,
        )
        result = post(outputs, sizes, sizes)
        return [
            {
                "prompt_index": i,
                "masks": r["masks_rle" if as_rle else "masks"],
                "scores": list(r["scores"]),
            }
            for i, r in enumerate(result)
        ]

    dense_results, rle_results = postprocess(False), postprocess(True)
    names = ["a", "b", "c"]
    prompts = [
        Sam3Prompt(type="text", text=name, output_prob_thresh=t)
        for name, t in zip(names, [0.3, 0.5, 0.7])
    ]
    image = SimpleNamespace(_read_shape_without_materialization=lambda: (64, 48))
    monkeypatch.setattr(v1_tensor, "_assemble_detections", lambda **kwargs: kwargs)
    monkeypatch.setattr(rle, "_assemble_detections", lambda **kwargs: kwargs)
    items = v2_tensor._collect_from_native_with_nms(
        dense_results, names, prompts, 0.3, apply_nms, 0.4
    )
    expected = v1_tensor._build_instance_detections(items, image, "rle")
    actual = rle.build_native_rle_detections(
        rle_results, names, None, prompts, 0.3, apply_nms, 0.4, image
    )
    for key in ["xyxy", "class_ids", "class_names_map"]:
        assert actual[key] == expected[key]
    np.testing.assert_array_equal(actual["confidences"], expected["confidences"])
    assert actual["mask"].masks == expected["mask"].masks
