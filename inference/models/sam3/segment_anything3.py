import hashlib
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import threading
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


def _estimate_numeric_bytes(obj, seen=None):
    """Estimate memory by summing bytes of numpy arrays and torch tensors found within obj.

    This intentionally ignores shallow sizes of non-numeric containers and objects, focusing on
    the dominant memory contributors for ML workloads.
    """
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    try:
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return int(obj.nbytes)
        # torch tensors (CPU or CUDA)
        if isinstance(obj, torch.Tensor):
            return int(obj.element_size() * obj.nelement())
    except Exception:
        # If any unexpected error occurs, skip sizing this object
        return 0

    # Recurse into common containers
    if isinstance(obj, dict):
        total = 0
        for key, value in obj.items():
            total += _estimate_numeric_bytes(key, seen)
            total += _estimate_numeric_bytes(value, seen)
        return total
    if isinstance(obj, (list, tuple, set, frozenset)):
        return sum(_estimate_numeric_bytes(item, seen) for item in obj)

    # Recurse into object attributes if available
    if hasattr(obj, "__dict__"):
        return _estimate_numeric_bytes(vars(obj), seen)
    if hasattr(obj, "__slots__"):
        total = 0
        for slot in obj.__slots__:
            try:
                total += _estimate_numeric_bytes(getattr(obj, slot), seen)
            except Exception:
                continue
        return total

    return 0


# its bith a core model and fine tuned model...
class SegmentAnything3(RoboflowCoreModel):
    """SAM3 wrapper with a similar interface to SAM2 in this codebase."""

    def __init__(
        self,
        *args,
        model_id: str = "sam3/paper_image_only_checkpoint_presence_0.35_completed_model_only",
        **kwargs,
    ):
        super().__init__(*args, model_id=model_id, **kwargs)
        # Lazy import SAM3 to avoid hard dependency when disabled
        # import sys
        # if SAM3_REPO_PATH not in sys.path:
        #     sys.path.append(SAM3_REPO_PATH)
        from sam3 import build_sam3_image_model

        model_version = model_id.split("/")[1]

        # base models have presence token if "presence is in then name
        # for fine tuned models right now at least its always false
        # we should add a config file to the model artifacts for this
        # has_presence_token = "presence" in model_id and model_id.startswith("sam3/")
        checkpoint = self.cache_file("weights.pt")
        bpe_path = self.cache_file("bpe_simple_vocab_16e6.txt.gz")

        # if model_version == "sam3_prod_v12_interactive_5box_image_only":
        #     has_presence_token = False

        self.sam3_lock = threading.RLock()

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            # has_presence_token=has_presence_token,
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
        self.postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes=True,
            convert_mask_to_rle=False,
            detection_threshold=0.5,
            to_cpu=False,
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

        model_version = self.endpoint.split("/")[1]

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
        with self.sam3_lock:
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
            start_ts = perf_counter()
            # Load image as PIL and build a SAM3 Datapoint with ordered prompts
            np_image = load_image_rgb(image)
            pil_image = Image.fromarray(np_image)
            h, w = pil_image.size[1], pil_image.size[0]

            datapoint = Sam3Datapoint(
                find_queries=[], images=[], raw_images=[pil_image]
            )
            # attach image
            datapoint.images = [Sam3ImageDP(data=pil_image, objects=[], size=(h, w))]

            # Map prompt_index -> coco_id to retrieve results later
            prompt_coco_ids: List[int] = []
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
                prompt_coco_ids.append(next_id)
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
                prompt_coco_ids.append(next_id)
                next_id += 1

            # Build prompts in order; ignore points for PCS
            prompts = prompts or []
            # Normalize prompts that may arrive as dicts from request.dict()
            normalized_prompts: List[Sam3Prompt] = []
            for p in prompts:
                if isinstance(p, Sam3Prompt):
                    normalized_prompts.append(p)
                else:
                    try:
                        normalized_prompts.append(Sam3Prompt(**p))
                    except Exception:
                        continue
            prompts = normalized_prompts
            if not prompts:
                # Backward compat: legacy fields in kwargs already normalized into prompts by validator
                pass
            for p in prompts:
                if p.boxes:
                    _add_visual(p.boxes, p.box_labels or [], p.text)
                else:
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

            # Postprocess to original size and build per-prompt results
            processed = self.postprocessor.process_results(output, batch.find_metadatas)
        if len(prompt_coco_ids) == 1:
            # Legacy single response
            preds = []
            if format in ["polygon", "json"]:
                polygons = processed[prompt_coco_ids[0]]["masks"]
                scores = processed[prompt_coco_ids[0]]["scores"]
                for poly, score in zip(polygons, scores):
                    preds.append(
                        Sam3SegmentationPrediction(
                            masks=[p.tolist() for p in poly],
                            confidence=float(score),
                            format="polygon",
                        )
                    )
                return Sam3SegmentationResponse(
                    time=perf_counter() - start_ts, predictions=preds
                )
            elif format == "rle":
                rles = processed[prompt_coco_ids[0]]["masks_rle"]
                scores = processed[prompt_coco_ids[0]]["scores"]
                for rle, score in zip(rles, scores):
                    preds.append(
                        Sam3SegmentationPrediction(
                            masks=rle, confidence=float(score), format="rle"
                        )
                    )
                return Sam3SegmentationResponse(
                    time=perf_counter() - start_ts, predictions=preds
                )
            elif format == "binary":
                # Return binary .npz content as in legacy path
                mask_stack = processed[prompt_coco_ids[0]].get("masks")
                if mask_stack is None:
                    mask_stack = processed[prompt_coco_ids[0]].get(
                        "masks_rle"
                    )  # not ideal
                binary_vector = BytesIO()
                np.savez_compressed(binary_vector, masks=np.array(mask_stack))
                binary_vector.seek(0)
                return binary_vector.getvalue()
            else:
                raise ValueError(f"Invalid format {format}")

        # Multi-prompt batch response
        prompt_results: List[Sam3PromptResult] = []
        for idx, coco_id in enumerate(prompt_coco_ids):
            echo = Sam3PromptEcho(
                prompt_index=idx,
                type=("visual" if prompts[idx].boxes else "text"),
                text=prompts[idx].text,
                num_boxes=(len(prompts[idx].boxes) if prompts[idx].boxes else 0),
            )
            preds: List[Sam3SegmentationPrediction] = []
            if format in ["polygon", "json"]:
                polygons = processed[coco_id]["masks"]
                scores = processed[coco_id]["scores"]
                for poly, score in zip(polygons, scores):
                    preds.append(
                        Sam3SegmentationPrediction(
                            masks=[p.tolist() for p in poly],
                            confidence=float(score),
                            format="polygon",
                        )
                    )
            elif format == "rle":
                rles = processed[coco_id]["masks_rle"]
                scores = processed[coco_id]["scores"]
                for rle, score in zip(rles, scores):
                    preds.append(
                        Sam3SegmentationPrediction(
                            masks=rle, confidence=float(score), format="rle"
                        )
                    )
            elif format == "binary":
                mask_stack = processed[coco_id].get("masks")
                if mask_stack is None:
                    mask_stack = processed[coco_id].get("masks_rle")
                # For batch response, we keep polygon/rle list even for binary request to keep schema consistent
                # If binary is strictly required per prompt, we could embed npz bytes per prompt item (heavier)
                for m in mask_stack:
                    preds.append(
                        Sam3SegmentationPrediction(
                            masks=m, confidence=1.0, format="polygon"
                        )
                    )
            else:
                raise ValueError(f"Invalid format {format}")
            prompt_results.append(
                Sam3PromptResult(prompt_index=idx, echo=echo, predictions=preds)
            )
        return Sam3BatchSegmentationResponse(
            time=perf_counter() - start_ts, prompt_results=prompt_results
        )

    def _results_to_response(
        self, masks: np.ndarray, scores: np.ndarray, start_ts: float
    ) -> Sam3SegmentationResponse:
        predictions: List[Sam3SegmentationPrediction] = []
        polygons = masks2multipoly(masks >= 0.5)
        for poly, score in zip(polygons, scores):
            predictions.append(
                Sam3SegmentationPrediction(
                    masks=[p.tolist() for p in poly],
                    confidence=float(score),
                    format="polygon",
                )
            )
        return Sam3SegmentationResponse(
            time=perf_counter() - start_ts, predictions=predictions
        )

    def _results_to_rle_response(
        self, masks: np.ndarray, scores: np.ndarray, start_ts: float
    ) -> Sam3SegmentationResponse:
        predictions: List[Sam3SegmentationPrediction] = []

        for mask, score in zip(masks, scores):
            # Apply same threshold as polygon format
            mask_binary = (mask >= 0.5).astype(np.uint8)

            # Encode mask to RLE format
            rle = mask_utils.encode(np.asfortranarray(mask_binary))
            rle["counts"] = rle["counts"].decode("utf-8")

            predictions.append(
                Sam3SegmentationPrediction(
                    masks=rle, confidence=float(score), format="rle"
                )
            )

        return Sam3SegmentationResponse(
            time=perf_counter() - start_ts, predictions=predictions
        )
