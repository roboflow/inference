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
)
from inference.core.entities.responses.sam3 import (
    Sam3EmbeddingResponse,
    Sam3SegmentationPrediction,
    Sam3SegmentationResponse,
)
from inference.core.env import SAM3_IMAGE_SIZE, SAM3_EMBEDDING_CACHE_SIZE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly


import sam3

print("sam3.__version__", sam3.__version__)


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
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3 import build_sam3_image_model

        model_version = model_id.split("/")[1]

        has_presence_token = True
        checkpoint = self.cache_file("weights.pt")
        bpe_path = self.cache_file("bpe_simple_vocab_16e6.txt.gz")

        if model_version == "sam3_prod_v12_interactive_5box_image_only":
            has_presence_token = False

        self.sam3_lock = threading.RLock()

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            has_presence_token=has_presence_token,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.processor = Sam3Processor()

        self.image_size = SAM3_IMAGE_SIZE
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache_keys: List[str] = []
        # Reasonable default since embeddings are heavy
        self.embedding_cache_size: int = SAM3_EMBEDDING_CACHE_SIZE
        self.task_type = "unsupervised-segmentation"

    # # Override to prevent Roboflow API/download flows during initialization
    # def download_weights(self) -> None:
    #     return None

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

    def _init_and_cache_state(
        self, image: Optional[InferenceRequestImage], image_id: Optional[str]
    ) -> Tuple[dict, str]:
        # Fast path if caller provided an id that is already cached
        if image_id and image_id in self.embedding_cache:
            return self.embedding_cache[image_id]["state"], image_id

        # load image as numpy array
        np_image = None
        if image is not None:
            np_image = load_image_rgb(image)
            generated_id = hashlib.md5(np_image.tobytes()).hexdigest()[:12]
            image_id = image_id or generated_id
        elif image_id is None:
            raise ValueError("Must provide either image or image_id")

        # If we computed or resolved an image_id above, check cache again to avoid recomputation
        if image_id in self.embedding_cache:
            return self.embedding_cache[image_id]["state"], image_id

        inference_state = self.processor(images=np_image)

        self.embedding_cache[image_id] = {"state": inference_state}
        # try:
        #     bytes_total = _estimate_numeric_bytes(inference_state)
        #     print(
        #         f"[SAM3] Cached inference_state for {image_id} ~ {bytes_total / (1024 * 1024):.2f} MB"
        #     )
        # except Exception as e:
        #     print(f"[SAM3] Failed to estimate size for {image_id}: {e}")

        # De-duplicate before appending to maintain order without duplicates
        if image_id in self.embedding_cache_keys:
            self.embedding_cache_keys.remove(image_id)
        self.embedding_cache_keys.append(image_id)
        if len(self.embedding_cache_keys) > self.embedding_cache_size:
            old = self.embedding_cache_keys.pop(0)
            # Use pop with default to avoid KeyError in rare race/out-of-sync cases
            self.embedding_cache.pop(old, None)
        return inference_state, image_id

    def embed_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        **kwargs,
    ):
        t1 = perf_counter()
        _, image_id = self._init_and_cache_state(image=image, image_id=image_id)
        return Sam3EmbeddingResponse(time=perf_counter() - t1, image_id=image_id)

    def infer_from_request(self, request: Sam3InferenceRequest):
        with self.sam3_lock:
            t1 = perf_counter()
            if isinstance(request, Sam3EmbeddingRequest):
                return self.embed_image(**request.dict())
            elif isinstance(request, Sam3SegmentationRequest):
                masks, scores = self.segment_image(**request.dict())
                # `json` is legacy default
                if request.format in ["polygon", "json"]:
                    return self._results_to_response(
                        masks=masks, scores=scores, start_ts=t1
                    )
                elif request.format == "rle":
                    return self._results_to_rle_response(
                        masks=masks, scores=scores, start_ts=t1
                    )
                elif request.format == "binary":
                    binary_vector = BytesIO()
                    np.savez_compressed(binary_vector, masks=masks)
                    binary_vector.seek(0)
                    return binary_vector.getvalue()
                else:
                    raise ValueError(f"Invalid format {request.format}")
            else:
                raise ValueError(f"Invalid request type {type(request)}")

    def segment_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        text: Optional[str] = None,
        points: Optional[List[List[float]]] = None,
        point_labels: Optional[List[int]] = None,
        boxes: Optional[List[List[float]]] = None,
        box_labels: Optional[List[int]] = None,
        instance_prompt: bool = False,
        output_prob_thresh: float = 0.5,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inference_state, image_id = self._init_and_cache_state(
            image=image, image_id=image_id
        )

        # Clear previous prompts
        self.processor.reset_state(inference_state)

        self.processor.add_prompt(
            inference_state,
            frame_idx=0,
            text_str=text,
            clear_old_points=True,
            points=points,
            point_labels=point_labels,
            boxes_xywh=boxes,
            box_labels=box_labels,
            clear_old_boxes=True,
            instance_prompt=instance_prompt,
        )

        self.model.run_inference(inference_state)

        outputs = self.processor.postprocess_output(
            inference_state, output_prob_thresh=output_prob_thresh
        )

        out_binary_masks = outputs["out_binary_masks"]  # (N, H, W) bool
        out_probs = outputs["out_probs"]  # (N,)

        masks = out_binary_masks.astype(np.uint8)
        scores = out_probs.astype(np.float32)
        return masks, scores

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
