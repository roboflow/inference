import hashlib
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import threading
import logging
from pycocotools import mask as mask_utils

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam3 import (
    Sam3EmbeddingRequest,
    Sam3InferenceRequest,
    Sam3SegmentationRequest,
    Sam3Prompt,
)
from inference.core.entities.responses.sam3 import (
    Sam3EmbeddingResponse,
    Sam3SegmentationPrediction,
    Sam3SegmentationResponse,
    Sam3BatchSegmentationResponse,
    Sam3PromptEcho,
    Sam3PromptResult,
)
from inference.core.env import (
    SAM3_IMAGE_SIZE,
    SAM3_EMBEDDING_CACHE_SIZE,
    MODELS_CACHE_AUTH_ENABLED,
    CORE_MODEL_BUCKET,
    INFER_BUCKET,
)
from inference.core.models.roboflow import (
    RoboflowCoreModel,
    is_model_artefacts_bucket_available,
)
from inference.core.cache.model_artifacts import (
    are_all_files_cached,
    save_bytes_in_cache,
)
from inference.core.exceptions import (
    ModelArtefactError,
    RoboflowAPINotAuthorizedError,
)
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_roboflow_model_data,
    get_from_url,
)
from inference.core.registries.roboflow import _check_if_api_key_has_access_to_model
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly


import sam3
from PIL import Image

print("sam3.__version__", sam3.__version__)

# SAM3 batched PCS utilities
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.train.data.sam3_image_dataset import (
    Datapoint as Sam3Datapoint,
    Image as Sam3ImageDP,
    FindQueryLoaded,
    InferenceMetadata,
)
from sam3.train.data.collator import collate_fn_api
from sam3.train.utils.train_utils import copy_data_to_device
from sam3.train.eval.postprocessors import PostProcessImage


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


# its bith a core model and fine tuned model...
class SegmentAnything3(RoboflowCoreModel):
    """SAM3 wrapper with a similar interface to SAM2 in this codebase."""

    def __init__(
        self,
        *args,
        model_id: str = "sam3/sam3_image_model_only",
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

    def _is_core_sam3_endpoint(self) -> bool:
        return isinstance(self.endpoint, str) and self.endpoint.startswith("sam3/")

    @property
    def model_artifact_bucket(self):
        # Use CORE bucket for base SAM3, standard INFER bucket for fine-tuned models
        return CORE_MODEL_BUCKET if self._is_core_sam3_endpoint() else INFER_BUCKET

    def _find_bpe_url(self, api_data: Dict[str, Any]) -> Optional[str]:
        # Best-effort discovery of BPE URL in ORT response payloads
        candidates: List[str] = []

        def _maybe_add(value: Any):
            if isinstance(value, str):
                candidates.append(value)

        # Common locations
        _maybe_add(api_data.get("bpe"))
        _maybe_add(api_data.get("bpe_url"))
        ort = api_data.get("ort") if isinstance(api_data, dict) else None
        if isinstance(ort, dict):
            _maybe_add(ort.get("bpe"))
            _maybe_add(ort.get("bpe_url"))
            env = ort.get("environment")
            if isinstance(env, dict):
                for k in ("sam3_bpe_url", "bpe_url", "bpe"):
                    _maybe_add(env.get(k))
        env = api_data.get("environment")
        if isinstance(env, dict):
            for k in ("sam3_bpe_url", "bpe_url", "bpe"):
                _maybe_add(env.get(k))

        # Prefer gzipped vocab files
        for url in candidates:
            if isinstance(url, str) and url.endswith(".gz"):
                return url
        # Fallback: any url-ish string containing "bpe"
        for url in candidates:
            if isinstance(url, str) and "bpe" in url.lower():
                return url
        return None

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
                resp = get_from_url(
                    url, json_response=False, verify_content_length=True
                )
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

    # Embedding cache removed for SAM3 PCS path; images processed per request

    def embed_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        **kwargs,
    ):
        # No-op or simple passthrough since we do not cache embeddings in PCS path
        t1 = perf_counter()
        computed_id = (
            image_id or hashlib.md5(load_image_rgb(image).tobytes()).hexdigest()[:12]
        )
        return Sam3EmbeddingResponse(time=perf_counter() - t1, image_id=computed_id)

    def infer_from_request(self, request: Sam3InferenceRequest):
        # with self.sam3_lock:
        t1 = perf_counter()
        if isinstance(request, Sam3EmbeddingRequest):
            return self.embed_image(**request.dict())
        elif isinstance(request, Sam3SegmentationRequest):
            result = self.segment_image(**request.dict())
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
        **kwargs,
    ):
        # Inference-only path; disable autograd throughout
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                start_ts = perf_counter()
                logging.debug(
                    f"SAM3.segment_image: start format={format} thresh={output_prob_thresh} prompts_in={(len(prompts) if prompts else 0)}"
                )
                # Load image as PIL and build a SAM3 Datapoint with ordered prompts
                np_image = load_image_rgb(image)
                pil_image = Image.fromarray(np_image)
                h, w = pil_image.size[1], pil_image.size[0]
                logging.debug(
                    f"SAM3.segment_image: np_image.shape={getattr(np_image, 'shape', None)} pil_size={(w, h)}"
                )

                datapoint = Sam3Datapoint(
                    find_queries=[], images=[], raw_images=[pil_image]
                )
                # attach image
                datapoint.images = [Sam3ImageDP(data=pil_image, objects=[], size=(h, w))]

                # Map prompt_index -> prompt_id to retrieve results later
                prompt_ids: List[int] = []
                next_id = 0

                def _add_text(text: Optional[str]):
                    nonlocal next_id
                    q = FindQueryLoaded(
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
                            coco_image_id=next_id,
                            original_image_id=next_id,
                            original_category_id=1,
                            original_size=(h, w),
                            object_id=0,
                            frame_index=0,
                        ),
                    )
                    datapoint.find_queries.append(q)
                    prompt_ids.append(next_id)
                    next_id += 1

                def _add_visual(
                    boxes_xywh_norm: List[List[float]],
                    labels: List[Union[int, bool]],
                    text: Optional[str],
                ):
                    nonlocal next_id
                    # Convert normalized XYWH -> pixel XYXY
                    xyxy: List[List[float]] = []
                    for x, y, bw, bh in boxes_xywh_norm or []:
                        x0 = x * w
                        y0 = y * h
                        x1 = x0 + bw * w
                        y1 = y0 + bh * h
                        xyxy.append([x0, y0, x1, y1])
                    labels_bool = [bool(int(v)) for v in (labels or [])]
                    q = FindQueryLoaded(
                        query_text=text if text is not None else "visual",
                        image_id=0,
                        object_ids_output=[],
                        is_exhaustive=True,
                        query_processing_order=0,
                        input_bbox=(
                            torch.tensor(xyxy, dtype=torch.float32) if xyxy else None
                        ),
                        input_bbox_label=(
                            torch.tensor(labels_bool, dtype=torch.bool)
                            if labels_bool
                            else None
                        ),
                        input_points=None,
                        semantic_target=None,
                        is_pixel_exhaustive=None,
                        inference_metadata=InferenceMetadata(
                            coco_image_id=next_id,
                            original_image_id=next_id,
                            original_category_id=1,
                            original_size=(h, w),
                            object_id=0,
                            frame_index=0,
                        ),
                    )
                    datapoint.find_queries.append(q)
                    prompt_ids.append(next_id)
                    next_id += 1

                # Build prompts in order; ignore points for PCS
                prompts = prompts or []
                # Normalize prompts that may arrive as dicts from request.dict()
                normalized_prompts: List[Sam3Prompt] = []
                for p in prompts:
                    if isinstance(p, Sam3Prompt):
                        normalized_prompts.append(p)
                    elif isinstance(p, dict):
                        try:
                            normalized_prompts.append(Sam3Prompt(**p))
                        except Exception as e:
                            logging.debug(
                                f"SAM3.segment_image: failed to normalize prompt dict: {e}"
                            )
                            continue
                prompts = normalized_prompts
                logging.debug(f"SAM3.segment_image: normalized_prompts={len(prompts)}")
                if not prompts:
                    # Backward compat: legacy fields in kwargs already normalized into prompts by validator
                    pass
                for p in prompts:
                    if p.boxes:
                        logging.debug(
                            f"SAM3.segment_image: add_visual boxes={len(p.boxes or [])} labels={len(p.box_labels or [])} text={p.text}"
                        )
                        _add_visual(p.boxes, p.box_labels or [], p.text)
                    else:
                        logging.debug(f"SAM3.segment_image: add_text text={p.text}")
                        _add_text(p.text)

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
                logging.debug("SAM3.segment_image: model forward done")

                # Postprocess to original size and build per-prompt results
                post = PostProcessImage(
                    max_dets_per_img=-1,
                    iou_type="segm",
                    use_original_sizes=True,
                    convert_mask_to_rle=False,
                    detection_threshold=float(
                        output_prob_thresh if output_prob_thresh is not None else 0.35
                    ),
                    to_cpu=True,
                )
                processed = post.process_results(output, batch.find_metadatas)
                logging.debug(
                    f"SAM3.segment_image: postprocess done; stages={len(processed)} ids_sample={list(processed.keys())[:3]}"
                )

        if len(prompt_ids) == 1:
            # Legacy single response
            masks_np = _to_numpy_masks(processed[prompt_ids[0]].get("masks"))
            scores = list(processed[prompt_ids[0]].get("scores", []))
            logging.debug(
                f"SAM3 single-prompt: masks_shape={getattr(masks_np, 'shape', None)} scores_len={len(scores)}"
            )
            preds = _masks_to_predictions(masks_np, scores, format)
            return Sam3SegmentationResponse(
                time=perf_counter() - start_ts, predictions=preds
            )

        # Multi-prompt batch response
        prompt_results: List[Sam3PromptResult] = []
        for idx, coco_id in enumerate(prompt_ids):
            echo = Sam3PromptEcho(
                prompt_index=idx,
                type=("visual" if prompts[idx].boxes else "text"),
                text=prompts[idx].text,
                num_boxes=(len(prompts[idx].boxes) if prompts[idx].boxes else 0),
            )
            masks_np = _to_numpy_masks(processed[coco_id].get("masks"))
            scores = list(processed[coco_id].get("scores", []))
            preds = _masks_to_predictions(masks_np, scores, format)
            prompt_results.append(
                Sam3PromptResult(prompt_index=idx, echo=echo, predictions=preds)
            )
        return Sam3BatchSegmentationResponse(
            time=perf_counter() - start_ts, prompt_results=prompt_results
        )
