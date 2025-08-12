import hashlib
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

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
from inference.core.env import SAM3_IMAGE_SIZE, SAM3_REPO_PATH
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly


class SegmentAnything3(RoboflowCoreModel):
    """SAM3 wrapper with a similar interface to SAM2 in this codebase."""

    def __init__(self, *args, model_id: str = "sam3", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        # Lazy import SAM3 to avoid hard dependency when disabled
        import sys
        if SAM3_REPO_PATH not in sys.path:
            sys.path.append(SAM3_REPO_PATH)
        from sam3 import build_sam3_image_model

        # if SAM3_CHECKPOINT_PATH is None:
        #     raise ValueError(
        #         "SAM3_CHECKPOINT_PATH must be set in environment to load SAM3 weights"
        #     )

        checkpoint = self.cache_file("sam3_prod_v12_interactive_5box_image_only.pt")
        bpe_path = self.cache_file("bpe_simple_vocab_16e6.txt.gz")

        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            checkpoint_path=checkpoint,
            device="cuda" if torch.cuda.is_available() else "cpu",
            eval_mode=True,
        )
        self.image_size = SAM3_IMAGE_SIZE
        self.embedding_cache: Dict[str, Dict[str, Any]] = {}
        self.embedding_cache_keys: List[str] = []
        # Reasonable default since embeddings are heavy
        self.embedding_cache_size: int = 32
        self.task_type = "unsupervised-segmentation"

    # # Override to prevent Roboflow API/download flows during initialization
    # def download_weights(self) -> None:
    #     return None

    def get_infer_bucket_file_list(self) -> List[str]:
        # SAM3 weights managed by env; no core bucket artifacts
        return ["sam3_prod_v12_interactive_5box_image_only.pt", "bpe_simple_vocab_16e6.txt.gz"]

    def preproc_image(self, image: InferenceRequestImage) -> np.ndarray:
        np_image = load_image_rgb(image)
        return np_image

    def _init_and_cache_state(
        self, image: Optional[InferenceRequestImage], image_id: Optional[str]
    ) -> Tuple[dict, str]:
        if image_id and image_id in self.embedding_cache:
            return self.embedding_cache[image_id]["state"], image_id

        np_image = None
        if image is not None:
            np_image = self.preproc_image(image)
            generated_id = hashlib.md5(np_image.tobytes()).hexdigest()[:12]
            image_id = image_id or generated_id
        elif image_id is None:
            raise ValueError("Must provide either image or image_id")

        # Save the image temporarily to a buffer path because SAM3 expects a file path
        # We keep it simple by using a temporary file in memory via PIL path-like not supported, so write to tmp
        from PIL import Image
        import tempfile
        import os

        if np_image is None:
            raise ValueError("image is required when state not cached")
        pil = Image.fromarray(np_image)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            pil.save(tmp.name, format="JPEG")
            tmp_path = tmp.name

        try:
            inference_state = self.model.init_state(tmp_path, offload_to_cpu=not torch.cuda.is_available())
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

        self.embedding_cache[image_id] = {"state": inference_state}
        self.embedding_cache_keys.append(image_id)
        if len(self.embedding_cache_keys) > self.embedding_cache_size:
            old = self.embedding_cache_keys.pop(0)
            del self.embedding_cache[old]
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
        t1 = perf_counter()
        if isinstance(request, Sam3EmbeddingRequest):
            return self.embed_image(**request.dict())
        elif isinstance(request, Sam3SegmentationRequest):
            masks, scores = self.segment_image(**request.dict())
            if request.format == "json":
                return self._results_to_response(
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
        inference_state, image_id = self._init_and_cache_state(image=image, image_id=image_id)

        # Clear previous prompts
        self.model.reset_state(inference_state)

        outputs = self.model.add_prompt(
            inference_state,
            frame_idx=0,
            text_str=text,
            clear_old_points=True,
            points=points,
            point_labels=point_labels,
            boxes_xywh=boxes,
            box_labels=box_labels,
            clear_old_boxes=True,
            output_prob_thresh=output_prob_thresh,
            instance_prompt=instance_prompt,
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
                    masks=[p.tolist() for p in poly], confidence=float(score)
                )
            )
        return Sam3SegmentationResponse(
            time=perf_counter() - start_ts, predictions=predictions
        )


