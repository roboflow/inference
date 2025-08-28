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
from inference.core.env import SAM3_IMAGE_SIZE, SAM3_EMBEDDING_CACHE_SIZE
from inference.core.models.roboflow import RoboflowCoreModel
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly


class SegmentAnything3(RoboflowCoreModel):
    """SAM3 wrapper with a similar interface to SAM2 in this codebase."""

    def __init__(self, *args, model_id: str = "sam3", **kwargs):
        super().__init__(*args, model_id=model_id, **kwargs)
        # Lazy import SAM3 to avoid hard dependency when disabled
        # import sys
        # if SAM3_REPO_PATH not in sys.path:
        #     sys.path.append(SAM3_REPO_PATH)
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
        self.embedding_cache_size: int = SAM3_EMBEDDING_CACHE_SIZE
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
        # Fast path if caller provided an id that is already cached
        if image_id and image_id in self.embedding_cache:
            return self.embedding_cache[image_id]["state"], image_id

        np_image = None
        if image is not None:
            np_image = self.preproc_image(image)
            generated_id = hashlib.md5(np_image.tobytes()).hexdigest()[:12]
            image_id = image_id or generated_id
        elif image_id is None:
            raise ValueError("Must provide either image or image_id")

        # If we computed or resolved an image_id above, check cache again to avoid recomputation
        if image_id in self.embedding_cache:
            return self.embedding_cache[image_id]["state"], image_id

        # Directly initialize inference_state from the in-memory image to avoid temp files,
        # mirroring the original SAM3 demo's init_state/load_image_as_single_frame logic.
        if np_image is None:
            raise ValueError("image is required when state not cached")

        try:
            image_size = getattr(self.model, "image_size", self.image_size)
            # Use the already-available NumPy image directly without round-tripping through PIL
            orig_h, orig_w = np_image.shape[:2]
            img_np = np_image
            if img_np.dtype == np.uint8:
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float32) / 255.0
            elif np.issubdtype(img_np.dtype, np.floating):
                img_t = torch.from_numpy(img_np).permute(2, 0, 1).to(dtype=torch.float32)
            else:
                raise RuntimeError(f"Unknown image dtype: {img_np.dtype}")

            # Resize to the model's expected square resolution
            img_t = img_t.unsqueeze(0)  # (1, C, H, W)
            img_t = torch.nn.functional.interpolate(
                img_t, size=(image_size, image_size), mode="bilinear", align_corners=False
            )
            images = img_t.half()

            img_mean_vals = getattr(self.model, "image_mean", (0.5, 0.5, 0.5))
            img_std_vals = getattr(self.model, "image_std", (0.5, 0.5, 0.5))
            img_mean = torch.tensor(img_mean_vals, dtype=torch.float16)[:, None, None]
            img_std = torch.tensor(img_std_vals, dtype=torch.float16)[:, None, None]

            if torch.cuda.is_available():
                images = images.cuda()
                img_mean = img_mean.cuda()
                img_std = img_std.cuda()

            images -= img_mean
            images /= img_std

            inference_state = {}
            inference_state["image_size"] = image_size
            inference_state["num_frames"] = len(images)
            inference_state["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            inference_state["orig_height"] = orig_h
            inference_state["orig_width"] = orig_w
            inference_state["constants"] = {}

            # Let the model construct its input batch and placeholders as in the original implementation
            self.model._construct_initial_input_batch(inference_state, images)
        except Exception:
            # Fallback to original temp-file pathway for safety if anything unexpected happens
            import tempfile
            import os
            from PIL import Image
            print("Falling back to temp file pathway")
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


