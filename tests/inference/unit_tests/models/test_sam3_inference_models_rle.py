"""Compressed SAM3 HTTP responses match the existing dense-mask API path."""

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pycocotools import mask as mask_utils

ADAPTER_MODULE = "inference.models.sam3.segment_anything3_inference_models"


@pytest.fixture(scope="module")
def adapter_module():
    # These tests exercise the real HTTP adapter and mask processing. Model
    # construction and the unrelated legacy SAM3 initializer need no weights.
    package = ModuleType("inference.models.sam3")
    package.__path__ = [
        str(Path(__file__).resolve().parents[4] / "inference" / "models" / "sam3")
    ]
    backend = ModuleType("inference_models.models.sam3.sam3_torch")
    backend.SAM3Torch = object
    with patch.dict(
        sys.modules,
        {
            "inference.models.sam3": package,
            "inference_models.models.sam3.sam3_torch": backend,
        },
    ):
        sys.modules.pop(ADAPTER_MODULE, None)
        yield importlib.import_module(ADAPTER_MODULE)


def _model_outputs(seed):
    rng = np.random.default_rng(seed)
    outputs = []
    for index in range(3):
        masks = (rng.random((5, 24, 32)) > 0.6).astype(np.uint8)
        masks[0] = 0  # Include an empty mask, which the API currently retains.
        masks[1:3] = 0
        masks[1:3, 3:15, 5:22] = 1  # Duplicate masks and tied scores exercise NMS.
        masks[2, 6:10, 9:14] = 0
        outputs.append(
            {"prompt_index": index, "masks": masks, "scores": [0.4, 0.8, 0.8, 0.6, 0.5]}
        )
    return outputs


def _dense_reference(module, outputs, prompts, nms):
    processed = {idx: result for idx, result in enumerate(outputs)}
    if nms is not None:
        masks = module._collect_masks_with_per_prompt_threshold(
            processed=processed, prompts=prompts, default_threshold=0.5
        )
        masks = module._apply_nms_cross_prompt(masks, nms)
        grouped = module._regroup_masks_by_prompt(masks, len(prompts))
    results = []
    for idx, prompt in enumerate(prompts):
        if nms is not None:
            bucket = grouped[idx]
            masks = np.stack([m for m, _ in bucket]) if bucket else np.zeros((0, 0, 0))
            scores = [s for _, s in bucket]
        else:
            masks, scores = outputs[idx]["masks"], outputs[idx]["scores"]
            if prompt.output_prob_thresh is not None:
                masks, scores = module._filter_by_threshold(
                    masks, scores, prompt.output_prob_thresh
                )
        results.append(
            module.Sam3PromptResult(
                prompt_index=idx,
                echo=module._build_echo(idx, prompt),
                predictions=module._masks_to_predictions(masks, scores, "rle"),
            ).model_dump()
        )
    return results


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("nms", [None, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("string_counts", [False, True])
def test_rle_http_response_matches_dense_reference_without_pixel_conversion(
    adapter_module, monkeypatch, seed, nms, string_counts
):
    module = adapter_module
    outputs = _model_outputs(seed)
    prompts = [
        module.Sam3Prompt(type="text", text="box"),
        module.Sam3Prompt(type="text", text="bottle", output_prob_thresh=0.4),
        module.Sam3Prompt(type="text", text="gap", output_prob_thresh=0.7),
    ]
    expected = _dense_reference(module, outputs, prompts, nms)
    compressed = []
    for result in outputs:
        rles = [mask_utils.encode(np.asfortranarray(mask)) for mask in result["masks"]]
        if string_counts:
            rles = [{**r, "counts": r["counts"].decode("ascii")} for r in rles]
        compressed.append({**result, "masks": rles})
    original = deepcopy(compressed)
    adapter = module.InferenceModelsSAM3Adapter.__new__(
        module.InferenceModelsSAM3Adapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_with_text_prompts.return_value = [compressed]
    monkeypatch.setattr(module, "load_image_rgb", lambda _: np.zeros((24, 32, 3)))

    def forbidden(*args, **kwargs):
        raise AssertionError("RLE response must not expand or re-encode mask pixels")

    monkeypatch.setattr(module, "_to_numpy_masks", forbidden)
    monkeypatch.setattr(module, "_masks_to_predictions", forbidden)
    monkeypatch.setattr(mask_utils, "encode", forbidden)
    monkeypatch.setattr(mask_utils, "decode", forbidden)
    response = adapter.segment_image(
        object(), prompts, format="rle", nms_iou_threshold=nms
    )

    assert [p.model_dump() for p in response.prompt_results] == expected
    assert compressed == original
    call = adapter._model.segment_with_text_prompts.call_args.kwargs
    assert call["mask_format"] == "rle"
    assert call["output_prob_thresh"] == 0.4
    assert call["max_detections"] == module.SAM3_MAX_DETECTIONS
    assert response.time >= 0


@pytest.mark.parametrize("prompt_count", [0, 1, 3])
def test_rle_response_keeps_empty_prompt_buckets(
    adapter_module, monkeypatch, prompt_count
):
    module = adapter_module
    prompts = [
        module.Sam3Prompt(type="text", text=f"class-{i}") for i in range(prompt_count)
    ]
    adapter = module.InferenceModelsSAM3Adapter.__new__(
        module.InferenceModelsSAM3Adapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_with_text_prompts.return_value = [
        [{"masks": [], "scores": []} for _ in prompts]
    ]
    monkeypatch.setattr(module, "load_image_rgb", lambda _: np.zeros((24, 32, 3)))
    result = adapter.segment_image(
        object(), prompts, format="rle", nms_iou_threshold=0.5
    )

    assert [p.prompt_index for p in result.prompt_results] == list(range(prompt_count))
    assert all(p.predictions == [] for p in result.prompt_results)


@pytest.mark.parametrize("fmt", ["polygon", "json"])
def test_polygon_responses_keep_dense_model_output(adapter_module, monkeypatch, fmt):
    module = adapter_module
    adapter = module.InferenceModelsSAM3Adapter.__new__(
        module.InferenceModelsSAM3Adapter
    )
    adapter._model = MagicMock()
    adapter._model.segment_with_text_prompts.return_value = [[_model_outputs(0)[0]]]
    monkeypatch.setattr(module, "load_image_rgb", lambda _: np.zeros((24, 32, 3)))
    result = adapter.segment_image(
        object(), [module.Sam3Prompt(type="text", text="box")], format=fmt
    )

    assert (
        adapter._model.segment_with_text_prompts.call_args.kwargs["mask_format"]
        == "dense"
    )
    assert all(p.format == "polygon" for p in result.prompt_results[0].predictions)
