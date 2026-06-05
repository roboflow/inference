from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from pycocotools import mask as mask_utils

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam3 import (
    Sam3InferenceRequest,
    Sam3Prompt,
    Sam3SegmentationRequest,
)
from inference.core.entities.responses.sam3 import (
    Sam3PromptEcho,
    Sam3PromptResult,
    Sam3SegmentationPrediction,
    Sam3SegmentationResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DEVICE,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly
from inference.usage_tracking.collector import usage_collector
from inference_models import AutoModel
from inference_models.models.sam3.sam3_torch import SAM3Torch

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class InferenceModelsSAM3Adapter(Model):
    """Adapter wrapping inference_models SAM3Torch for open-vocabulary segmentation.

    Replaces inference.models.sam3.segment_anything3.SegmentAnything3.
    Handles Sam3SegmentationRequest with text and/or visual (box) prompts via
    SAM3Torch.segment_with_text_prompts.
    """

    def __init__(
        self,
        *args,
        model_id: str = "sam3/sam3_final",
        api_key: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        self.task_type = "unsupervised-segmentation"

        extra_weights_provider_headers = get_extra_weights_provider_headers(
            countinference=kwargs.get("countinference"),
            service_secret=kwargs.get("service_secret"),
        )
        backend = list(
            VALID_INFERENCE_MODELS_BACKENDS.difference(
                DISABLED_INFERENCE_MODELS_BACKENDS
            )
        )
        self._model: SAM3Torch = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=self.api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )

    @usage_collector("model")
    def infer_from_request(self, request: Sam3InferenceRequest):
        t1 = perf_counter()
        if isinstance(request, Sam3SegmentationRequest):
            return self.segment_image(
                image=request.image,
                prompts=request.prompts,
                output_prob_thresh=request.output_prob_thresh or 0.5,
                format=request.format or "polygon",
                nms_iou_threshold=request.nms_iou_threshold,
                inference_start_timestamp=t1,
            )
        raise ValueError(f"Invalid request type {type(request)}")

    def segment_image(
        self,
        image: InferenceRequestImage,
        prompts: List[Sam3Prompt],
        output_prob_thresh: float = 0.5,
        format: str = "polygon",
        nms_iou_threshold: Optional[float] = None,
        inference_start_timestamp: Optional[float] = None,
    ) -> Sam3SegmentationResponse:
        if inference_start_timestamp is None:
            inference_start_timestamp = perf_counter()
        np_image = load_image_rgb(image)

        # The backend applies a single threshold floor; use the min so per-prompt
        # thresholds applied below can still refine higher values.
        min_threshold = output_prob_thresh
        for p in prompts:
            prompt_thresh = getattr(p, "output_prob_thresh", None)
            if prompt_thresh is not None:
                min_threshold = min(min_threshold, prompt_thresh)

        prompt_dicts = [_sam3_prompt_to_dict(p) for p in prompts]

        # segment_with_text_prompts returns List[per-image] of List[per-prompt] dicts
        # with keys: prompt_index, masks (N,H,W ndarray), scores (list).
        per_image_results = self._model.segment_with_text_prompts(
            images=[np_image],
            prompts=prompt_dicts,
            output_prob_thresh=float(min_threshold),
        )
        per_prompt = per_image_results[0]

        # processed: prompt_idx -> {"masks": ndarray, "scores": list}
        processed: Dict[int, Dict[str, Any]] = {}
        for idx, r in enumerate(per_prompt):
            processed[idx] = {
                "masks": r.get("masks"),
                "scores": list(r.get("scores", [])),
            }

        if nms_iou_threshold is not None and len(prompts) > 0:
            all_masks = _collect_masks_with_per_prompt_threshold(
                processed=processed,
                prompts=prompts,
                default_threshold=output_prob_thresh,
            )
            if len(all_masks) > 0:
                all_masks = _apply_nms_cross_prompt(all_masks, nms_iou_threshold)
            regrouped = _regroup_masks_by_prompt(all_masks, len(prompts))

            prompt_results: List[Sam3PromptResult] = []
            for idx, p in enumerate(prompts):
                echo = _build_echo(idx, p)
                bucket = regrouped.get(idx, [])
                if bucket:
                    masks_np = np.stack([m for m, _ in bucket], axis=0)
                    scores = [s for _, s in bucket]
                else:
                    masks_np = np.zeros((0, 0, 0), dtype=np.uint8)
                    scores = []
                preds = _masks_to_predictions(masks_np, scores, format)
                prompt_results.append(
                    Sam3PromptResult(prompt_index=idx, echo=echo, predictions=preds)
                )
        else:
            prompt_results = []
            for idx, p in enumerate(prompts):
                masks_np = _to_numpy_masks(processed[idx]["masks"])
                scores = processed[idx]["scores"]
                prompt_thresh = getattr(p, "output_prob_thresh", None)
                if prompt_thresh is not None:
                    masks_np, scores = _filter_by_threshold(
                        masks_np, scores, prompt_thresh
                    )
                preds = _masks_to_predictions(masks_np, scores, format)
                prompt_results.append(
                    Sam3PromptResult(
                        prompt_index=idx,
                        echo=_build_echo(idx, p),
                        predictions=preds,
                    )
                )

        return Sam3SegmentationResponse(
            time=perf_counter() - inference_start_timestamp,
            prompt_results=prompt_results,
        )


def _sam3_prompt_to_dict(p: Sam3Prompt) -> Dict[str, Any]:
    d: Dict[str, Any] = {"text": p.text}
    if p.boxes:
        d["boxes"] = (
            p.boxes
        )  # backend's _build_visual_query handles pydantic Box/BoxXYXY
        d["box_labels"] = p.box_labels or []
    return d


def _build_echo(prompt_index: int, p: Sam3Prompt) -> Sam3PromptEcho:
    has_visual = bool(p.boxes)
    return Sam3PromptEcho(
        prompt_index=prompt_index,
        type="visual" if has_visual else "text",
        text=p.text,
        num_boxes=len(p.boxes) if has_visual else 0,
    )


def _to_numpy_masks(masks_any) -> np.ndarray:
    if masks_any is None:
        return np.zeros((0, 0, 0), dtype=np.uint8)
    if hasattr(masks_any, "detach"):
        masks_np = masks_any.detach().cpu().numpy().astype(np.uint8)
    else:
        arrs = []
        for m in masks_any:
            if hasattr(m, "detach"):
                arrs.append(m.detach().cpu().numpy().astype(np.uint8))
            else:
                arrs.append(np.asarray(m, dtype=np.uint8))
        if not arrs:
            return np.zeros((0, 0, 0), dtype=np.uint8)
        masks_np = np.stack(arrs, axis=0)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, ...]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]
    return masks_np


def _filter_by_threshold(
    masks_np: np.ndarray,
    scores: List[float],
    threshold: float,
) -> Tuple[np.ndarray, List[float]]:
    if masks_np.ndim != 3 or masks_np.shape[0] == 0:
        return masks_np, scores
    keep = [i for i, s in enumerate(scores) if s >= threshold]
    if not keep:
        return np.zeros((0, 0, 0), dtype=np.uint8), []
    return masks_np[keep], [scores[i] for i in keep]


def _masks_to_predictions(
    masks_np: np.ndarray, scores: List[float], fmt: str
) -> List[Sam3SegmentationPrediction]:
    preds: List[Sam3SegmentationPrediction] = []
    if masks_np.ndim != 3 or 0 in masks_np.shape:
        return preds
    if fmt in ("polygon", "json"):
        polygons = masks2multipoly((masks_np > 0).astype(np.uint8))
        for poly, score in zip(polygons, scores[: len(polygons)]):
            preds.append(
                Sam3SegmentationPrediction(
                    masks=[p.tolist() for p in poly],
                    confidence=float(score),
                    format="polygon",
                )
            )
    elif fmt == "rle":
        for m, score in zip(masks_np, scores[: masks_np.shape[0]]):
            mb = (m > 0).astype(np.uint8)
            rle = mask_utils.encode(np.asfortranarray(mb))
            rle["counts"] = rle["counts"].decode("utf-8")
            preds.append(
                Sam3SegmentationPrediction(
                    masks=rle, confidence=float(score), format="rle"
                )
            )
    return preds


def _collect_masks_with_per_prompt_threshold(
    processed: Dict[int, Dict[str, Any]],
    prompts: List[Sam3Prompt],
    default_threshold: float,
) -> List[Tuple[int, np.ndarray, float]]:
    all_masks: List[Tuple[int, np.ndarray, float]] = []
    for idx, p in enumerate(prompts):
        prompt_thresh = getattr(p, "output_prob_thresh", None)
        if prompt_thresh is None:
            prompt_thresh = default_threshold
        masks_np = _to_numpy_masks(processed[idx]["masks"])
        scores = processed[idx]["scores"]
        if masks_np.ndim != 3 or 0 in masks_np.shape:
            continue
        for mask, score in zip(masks_np, scores):
            if score >= prompt_thresh:
                all_masks.append((idx, mask, float(score)))
    return all_masks


def _nms_greedy_pycocotools(
    rles: List[Dict],
    confidences: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    num_detections = len(rles)
    if num_detections == 0:
        return np.array([], dtype=bool)
    sort_index = np.argsort(confidences)[::-1]
    sorted_rles = [rles[i] for i in sort_index]
    ious = mask_utils.iou(sorted_rles, sorted_rles, [0] * num_detections)
    keep = np.ones(num_detections, dtype=bool)
    for i in range(num_detections):
        if keep[i]:
            condition = ious[i, :] > iou_threshold
            keep[i + 1 :] = np.where(condition[i + 1 :], False, keep[i + 1 :])
    return keep[np.argsort(sort_index)]


def _apply_nms_cross_prompt(
    all_masks: List[Tuple[int, np.ndarray, float]],
    iou_threshold: float,
) -> List[Tuple[int, np.ndarray, float]]:
    if not all_masks:
        return all_masks
    rles = []
    for _, mask_np, _ in all_masks:
        mb = (mask_np > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mb))
        rles.append(rle)
    confidences = np.array([score for _, _, score in all_masks])
    keep = _nms_greedy_pycocotools(rles, confidences, iou_threshold)
    return [all_masks[i] for i in range(len(all_masks)) if keep[i]]


def _regroup_masks_by_prompt(
    filtered_masks: List[Tuple[int, np.ndarray, float]],
    num_prompts: int,
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    result: Dict[int, List[Tuple[np.ndarray, float]]] = {
        i: [] for i in range(num_prompts)
    }
    for prompt_idx, mask_np, score in filtered_masks:
        result[prompt_idx].append((mask_np, score))
    return result
