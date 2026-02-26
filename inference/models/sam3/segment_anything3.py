import hashlib
import logging
import threading
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import sam3
import torch
from PIL import Image
from pycocotools import mask as mask_utils

# from sam3.train.eval.postprocessors import PostProcessImage
from sam3.eval.postprocessors import PostProcessImage

# from sam3.train.utils.misc import copy_data_to_device
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.data.collator import collate_fn_api
from sam3.train.data.sam3_image_dataset import Datapoint as Sam3Datapoint
from sam3.train.data.sam3_image_dataset import FindQueryLoaded
from sam3.train.data.sam3_image_dataset import Image as Sam3ImageDP
from sam3.train.data.sam3_image_dataset import InferenceMetadata

# SAM3 batched PCS utilities
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    NormalizeAPI,
    RandomResizeAPI,
    ToTensorAPI,
)

from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    save_bytes_in_cache,
)
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
    CORE_MODEL_BUCKET,
    INFER_BUCKET,
    MODELS_CACHE_AUTH_ENABLED,
    SAM3_IMAGE_SIZE,
)
from inference.core.exceptions import ModelArtefactError, RoboflowAPINotAuthorizedError
from inference.core.models.roboflow import (
    RoboflowCoreModel,
    is_model_artefacts_bucket_available,
)
from inference.core.registries.roboflow import _check_if_api_key_has_access_to_model
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_from_url,
    get_roboflow_model_data,
)
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly
from inference.usage_tracking.collector import usage_collector


def _to_numpy_masks(masks_any) -> np.ndarray:
    """Convert masks from torch/list to numpy uint8 array (N,H,W).

    Automatically normalizes shape:
    - (N,1,H,W) -> (N,H,W)
    - (H,W) -> (1,H,W)
    - (N,H,W) -> unchanged
    """
    if masks_any is None:
        return np.zeros((0, 0, 0), dtype=np.uint8)

    # Convert to numpy
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

    # Normalize shape to (N,H,W)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0, ...]
    elif masks_np.ndim == 2:
        masks_np = masks_np[None, ...]

    return masks_np


def _masks_to_predictions(
    masks_np: np.ndarray, scores: List[float], fmt: str
) -> List[Sam3SegmentationPrediction]:
    """Convert boolean masks (N,H,W) to API predictions in requested format.

    Assumes masks_np is already normalized to (N,H,W) by _to_numpy_masks.

    Args:
        masks_np: Boolean or uint8 masks array (N,H,W)
        scores: Confidence scores per mask
        fmt: Output format: 'polygon', 'json', or 'rle'

    Returns:
        List of Sam3SegmentationPrediction
    """
    preds = []

    if masks_np.ndim != 3 or 0 in masks_np.shape:
        return preds

    if fmt in ["polygon", "json"]:
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


def _nms_greedy_pycocotools(
    rles: List[Dict],
    confidences: np.ndarray,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """
    NMS (Non-Maximum Suppression): removes duplicate detections by keeping only
    the highest confidence detection when multiple detections overlap the same object.

    Args:
        rles: List of RLE dictionaries
        confidences: Array of confidence scores
        iou_threshold: IoU threshold above which detections are suppressed

    Returns:
        Boolean array indicating which detections to keep (in original order)
    """
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
    """
    Apply NMS across all prompts.

    Args:
        all_masks: List of tuples (prompt_idx, mask_np, score)
        iou_threshold: IoU threshold for suppression

    Returns:
        Filtered list of (prompt_idx, mask_np, score) tuples
    """
    if not all_masks or len(all_masks) == 0:
        return all_masks

    # Convert masks to RLE for IoU calculation
    # We perform NMS using RLE to save memory (using masks could cause OOM)
    rles = []
    for _, mask_np, _ in all_masks:
        mb = (mask_np > 0).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mb))
        rles.append(rle)

    confidences = np.array([float(score) for _, _, score in all_masks])
    keep_indices = _nms_greedy_pycocotools(rles, confidences, iou_threshold)

    return [all_masks[i] for i in range(len(all_masks)) if keep_indices[i]]


def _collect_masks_with_per_prompt_threshold(
    processed: Dict[int, Dict],
    prompt_ids: List[int],
    prompts: List[Any],
    default_threshold: float,
) -> List[Tuple[int, np.ndarray, float]]:
    """
    Collect all masks applying per-prompt thresholds.

    Args:
        processed: Dict mapping coco_id -> {"masks": ..., "scores": ...}
        prompt_ids: List of prompt indices
        prompts: List of Sam3Prompt objects
        default_threshold: Default threshold if prompt doesn't specify one

    Returns:
        List of (prompt_idx, mask_np, score) tuples that pass threshold
    """
    all_masks = []

    for idx, coco_id in enumerate(prompt_ids):
        prompt_thresh = getattr(prompts[idx], "output_prob_thresh", None)
        if prompt_thresh is None:
            prompt_thresh = default_threshold

        masks_np = _to_numpy_masks(processed[coco_id].get("masks"))
        scores = list(processed[coco_id].get("scores", []))

        if masks_np.ndim != 3 or 0 in masks_np.shape:
            continue

        for mask_i, (mask, score) in enumerate(zip(masks_np, scores)):
            if score >= prompt_thresh:
                all_masks.append((idx, mask, score))

    return all_masks


def _regroup_masks_by_prompt(
    filtered_masks: List[Tuple[int, np.ndarray, float]],
    num_prompts: int,
) -> Dict[int, List[Tuple[np.ndarray, float]]]:
    """
    Regroup filtered masks back to per-prompt results.

    Args:
        filtered_masks: List of (prompt_idx, mask_np, score) tuples
        num_prompts: Total number of prompts

    Returns:
        Dict mapping prompt_idx -> list of (mask_np, score) tuples
    """
    result: Dict[int, List[Tuple[np.ndarray, float]]] = {
        i: [] for i in range(num_prompts)
    }

    for prompt_idx, mask_np, score in filtered_masks:
        result[prompt_idx].append((mask_np, score))

    return result


def _build_text_query(
    coco_id: int,
    h: int,
    w: int,
    text: Optional[str],
) -> FindQueryLoaded:
    """Create a FindQueryLoaded for a text-only prompt."""
    return FindQueryLoaded(
        query_text=text if text is not None else "visual",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=None,
        input_bbox_label=None,
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=coco_id,
            original_image_id=coco_id,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )


def _build_visual_query(
    coco_id: int,
    h: int,
    w: int,
    boxes: Optional[List[Any]],
    labels: Optional[List[Union[int, bool]]],
    text: Optional[str],
) -> FindQueryLoaded:
    """Create a FindQueryLoaded for a visual (box) prompt from absolute boxes.

    Accepts boxes as either XYWH (x,y,width,height) or XYXY (x0,y0,x1,y1) objects.
    """
    xyxy_pixels: List[List[float]] = []
    for b in boxes or []:
        if hasattr(b, "x"):
            x0 = float(b.x)
            y0 = float(b.y)
            x1 = x0 + float(b.width)
            y1 = y0 + float(b.height)
        else:
            x0 = float(b.x0)
            y0 = float(b.y0)
            x1 = float(b.x1)
            y1 = float(b.y1)
        xyxy_pixels.append([x0, y0, x1, y1])

    labels_bool = [bool(int(v)) for v in (labels or [])]

    return FindQueryLoaded(
        query_text=text if text is not None else "visual",
        image_id=0,
        object_ids_output=[],
        is_exhaustive=True,
        query_processing_order=0,
        input_bbox=(
            torch.tensor(xyxy_pixels, dtype=torch.float32) if xyxy_pixels else None
        ),
        input_bbox_label=(
            torch.tensor(labels_bool, dtype=torch.bool) if labels_bool else None
        ),
        input_points=None,
        semantic_target=None,
        is_pixel_exhaustive=None,
        inference_metadata=InferenceMetadata(
            coco_image_id=coco_id,
            original_image_id=coco_id,
            original_category_id=1,
            original_size=(h, w),
            object_id=0,
            frame_index=0,
        ),
    )


# its bith a core model and fine tuned model...
class SegmentAnything3(RoboflowCoreModel):
    """SAM3 wrapper with a similar interface to SAM2 in this codebase."""

    def __init__(
        self,
        *args,
        model_id: str = "sam3/sam3_final",
        **kwargs,
    ):
        super().__init__(*args, model_id=model_id, **kwargs)

        # Lazy import SAM3 to avoid hard dependency when disabled
        from sam3 import build_sam3_image_model

        checkpoint = self.cache_file("weights.pt")
        bpe_path = self.cache_file("bpe_simple_vocab_16e6.txt.gz")

        self.sam3_lock = threading.RLock()

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            load_from_HF=False,
            compile=False,
        )

        # Preprocessing and postprocessing for PCS image path
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=SAM3_IMAGE_SIZE,
                    max_size=SAM3_IMAGE_SIZE,
                    square=True,
                    consistent_transform=False,
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.image_size = SAM3_IMAGE_SIZE
        self.task_type = "unsupervised-segmentation"

    def warmup(self) -> None:
        """Run a dummy forward pass to trigger CUDA kernel JIT compilation."""
        logger = logging.getLogger(__name__)
        logger.info("SAM3 warmup: running preflight inference to compile CUDA kernels...")
        try:
            dummy_size = 256
            dummy_image = Image.new("RGB", (dummy_size, dummy_size), color=(128, 128, 128))

            with torch.inference_mode():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    datapoint = Sam3Datapoint(
                        find_queries=[],
                        images=[Sam3ImageDP(data=dummy_image, objects=[], size=(dummy_size, dummy_size))],
                    )
                    datapoint.find_queries.append(
                        _build_text_query(coco_id=0, h=dummy_size, w=dummy_size, text="warmup")
                    )

                    datapoint = self.transform(datapoint)
                    batch = collate_fn_api(batch=[datapoint], dict_key="dummy")["dummy"]
                    batch = copy_data_to_device(
                        batch,
                        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        non_blocking=True,
                    )

                    _ = self.model(batch)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            logger.info("SAM3 warmup: preflight inference completed successfully.")
        except Exception as e:
            logger.warning(f"SAM3 warmup: preflight inference failed (non-fatal): {e}")

    def _is_core_sam3_endpoint(self) -> bool:
        return isinstance(self.endpoint, str) and self.endpoint.startswith("sam3/")

    @property
    def model_artifact_bucket(self):
        # Use CORE bucket for base SAM3, standard INFER bucket for fine-tuned models
        return CORE_MODEL_BUCKET if self._is_core_sam3_endpoint() else INFER_BUCKET

    def download_weights(self) -> None:
        infer_bucket_files = self.get_infer_bucket_file_list()

        # Auth check aligned with chosen endpoint type
        if MODELS_CACHE_AUTH_ENABLED:
            endpoint_type = (
                ModelEndpointType.CORE_MODEL
                if self._is_core_sam3_endpoint()
                else ModelEndpointType.ORT
            )
            if not _check_if_api_key_has_access_to_model(
                api_key=self.api_key,
                model_id=self.endpoint,
                endpoint_type=endpoint_type,
            ):
                raise RoboflowAPINotAuthorizedError(
                    f"API key {self.api_key} does not have access to model {self.endpoint}"
                )

        # Already cached
        if are_all_files_cached(files=infer_bucket_files, model_id=self.endpoint):
            return None

        # S3 path works for both; keys are {endpoint}/<file>
        if is_model_artefacts_bucket_available():
            self.download_model_artefacts_from_s3()
            return None

        # API fallback
        if self._is_core_sam3_endpoint():
            # Base SAM3 from core_model endpoint; preserves filenames
            return super().download_model_from_roboflow_api()

        # Fine-tuned SAM3: use ORT endpoint to fetch weights map or model url
        api_data = get_roboflow_model_data(
            api_key=self.api_key,
            model_id=self.endpoint,
            endpoint_type=ModelEndpointType.ORT,
            device_id=self.device_id,
        )

        ort = api_data.get("ort") if isinstance(api_data, dict) else None
        if not isinstance(ort, dict):
            raise ModelArtefactError("ORT response malformed for fine-tuned SAM3")

        # Preferred: explicit weights map of filename -> URL
        weights_map = ort.get("weights")
        if isinstance(weights_map, dict) and len(weights_map) > 0:
            for filename, url in weights_map.items():
                resp = get_from_url(url, json_response=False)
                save_bytes_in_cache(
                    content=resp.content,
                    file=str(filename),
                    model_id=self.endpoint,
                )
            return None

        raise ModelArtefactError(
            "ORT response missing both 'weights' for fine-tuned SAM3"
        )

    def get_infer_bucket_file_list(self) -> List[str]:
        # SAM3 weights managed by env; no core bucket artifacts

        return [
            "weights.pt",
            "bpe_simple_vocab_16e6.txt.gz",
        ]

    def preproc_image(self, image: InferenceRequestImage) -> np.ndarray:
        np_image = load_image_rgb(image)
        return np_image

    @usage_collector("model")
    def infer_from_request(self, request: Sam3InferenceRequest):
        # with self.sam3_lock:
        t1 = perf_counter()
        if isinstance(request, Sam3SegmentationRequest):
            # Pass strongly-typed fields to preserve Sam3Prompt objects
            result = self.segment_image(
                image=request.image,
                image_id=request.image_id,
                prompts=request.prompts,
                output_prob_thresh=request.output_prob_thresh or 0.5,
                format=request.format or "polygon",
                nms_iou_threshold=request.nms_iou_threshold,
            )
            # segment_image now returns either bytes or a response model
            return result
        else:
            raise ValueError(f"Invalid request type {type(request)}")

    def segment_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        prompts: Optional[List[Sam3Prompt]] = None,
        output_prob_thresh: float = 0.5,
        format: Optional[str] = "polygon",
        nms_iou_threshold: Optional[float] = None,
        **kwargs,
    ):
        np_image = load_image_rgb(image)
        h, w = np_image.shape[:2]
        pil_image = Image.fromarray(np_image)

        # Inference-only path; disable autograd throughout
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                start_ts = perf_counter()

                # TODO this can also take tensor directly instead of PIL image, so we want to avoid double conversion
                # TODO: this also supports multiple images for multi batch inference
                datapoint = Sam3Datapoint(
                    find_queries=[],
                    images=[Sam3ImageDP(data=pil_image, objects=[], size=(h, w))],
                )

                # Build prompts in order
                prompts = prompts or []

                # Map prompt_index -> prompt_id to retrieve results later
                prompt_ids: List[int] = []
                for idx, p in enumerate(prompts):
                    if getattr(p, "boxes", None):
                        q = _build_visual_query(
                            coco_id=idx,
                            h=h,
                            w=w,
                            boxes=p.boxes,
                            labels=p.box_labels or [],
                            text=p.text,
                        )
                    else:
                        q = _build_text_query(
                            coco_id=idx,
                            h=h,
                            w=w,
                            text=p.text,
                        )
                    datapoint.find_queries.append(q)
                    prompt_ids.append(idx)

                # Transform and collate to BatchedDatapoint
                datapoint = self.transform(datapoint)
                batch = collate_fn_api(batch=[datapoint], dict_key="dummy")["dummy"]
                batch = copy_data_to_device(
                    batch,
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                    non_blocking=True,
                )

                # Forward
                output = self.model(batch)

                # Calculate minimum threshold for initial filtering
                # (we'll apply per-prompt thresholds later)
                min_threshold = output_prob_thresh
                for p in prompts:
                    prompt_thresh = getattr(p, "output_prob_thresh", None)
                    if prompt_thresh is not None:
                        min_threshold = min(min_threshold, prompt_thresh)

                # Postprocess to original size and build per-prompt results
                post = PostProcessImage(
                    max_dets_per_img=-1,
                    iou_type="segm",
                    use_original_sizes_box=True,
                    use_original_sizes_mask=True,
                    convert_mask_to_rle=False,
                    detection_threshold=float(
                        min_threshold if min_threshold is not None else 0.35
                    ),
                    to_cpu=True,
                )
                processed = post.process_results(output, batch.find_metadatas)

        needs_cross_prompt_nms = nms_iou_threshold is not None
        prompt_results: List[Sam3PromptResult] = []

        if needs_cross_prompt_nms and len(prompts) > 0:
            all_masks = _collect_masks_with_per_prompt_threshold(
                processed=processed,
                prompt_ids=prompt_ids,
                prompts=prompts,
                default_threshold=output_prob_thresh,
            )

            if len(all_masks) > 0:
                all_masks = _apply_nms_cross_prompt(all_masks, nms_iou_threshold)

            regrouped = _regroup_masks_by_prompt(all_masks, len(prompts))

            # Build prompt results from regrouped masks
            for idx, coco_id in enumerate(prompt_ids):
                has_visual = bool(getattr(prompts[idx], "boxes", None))
                num_boxes = len(prompts[idx].boxes or []) if has_visual else 0
                echo = Sam3PromptEcho(
                    prompt_index=idx,
                    type=("visual" if has_visual else "text"),
                    text=prompts[idx].text,
                    num_boxes=num_boxes,
                )

                # Convert regrouped masks to predictions
                prompt_masks = regrouped.get(idx, [])
                if prompt_masks:
                    masks_np = np.stack([m for m, _ in prompt_masks], axis=0)
                    scores = [s for _, s in prompt_masks]
                else:
                    masks_np = np.zeros((0, 0, 0), dtype=np.uint8)
                    scores = []

                preds = _masks_to_predictions(masks_np, scores, format)
                prompt_results.append(
                    Sam3PromptResult(prompt_index=idx, echo=echo, predictions=preds)
                )
        else:
            for idx, coco_id in enumerate(prompt_ids):
                has_visual = bool(getattr(prompts[idx], "boxes", None))
                num_boxes = len(prompts[idx].boxes or []) if has_visual else 0
                echo = Sam3PromptEcho(
                    prompt_index=idx,
                    type=("visual" if has_visual else "text"),
                    text=prompts[idx].text,
                    num_boxes=num_boxes,
                )
                masks_np = _to_numpy_masks(processed[coco_id].get("masks"))
                scores = list(processed[coco_id].get("scores", []))
                prompt_thresh = getattr(prompts[idx], "output_prob_thresh", None)
                if prompt_thresh is not None:
                    masks_np, scores = _filter_by_threshold(
                        masks_np, scores, prompt_thresh
                    )
                preds = _masks_to_predictions(masks_np, scores, format)
                prompt_results.append(
                    Sam3PromptResult(prompt_index=idx, echo=echo, predictions=preds)
                )

        return Sam3SegmentationResponse(
            time=perf_counter() - start_ts, prompt_results=prompt_results
        )
