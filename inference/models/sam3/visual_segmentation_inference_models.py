import copy
from io import BytesIO
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from pycocotools import mask as mask_utils

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam2 import (
    Sam2EmbeddingRequest,
    Sam2InferenceRequest,
    Sam2PromptSet,
    Sam2SegmentationRequest,
)
from inference.core.entities.responses.sam2 import (
    Sam2EmbeddingResponse,
    Sam2SegmentationPrediction,
    Sam2SegmentationResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    API_KEY,
    DEVICE,
    DISABLE_SAM3_LOGITS_CACHE,
    DISABLED_INFERENCE_MODELS_BACKENDS,
    SAM3_MAX_EMBEDDING_CACHE_SIZE,
    SAM3_MAX_LOGITS_CACHE_SIZE,
    VALID_INFERENCE_MODELS_BACKENDS,
)
from inference.core.models.base import Model
from inference.core.roboflow_api import get_extra_weights_provider_headers
from inference.core.utils.image_utils import load_image_rgb
from inference.core.utils.postprocess import masks2multipoly
from inference.usage_tracking.collector import usage_collector
from inference_models import AutoModel
from inference_models.models.sam3.cache import (
    Sam3ImageEmbeddingsInMemoryCache,
    Sam3LowResolutionMasksInMemoryCache,
)
from inference_models.models.sam3.sam3_torch import SAM3Torch

if DEVICE is None:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MASK_THRESHOLD = 0.0


class InferenceModelsSAM3InteractiveAdapter(Model):
    """Adapter wrapping inference_models SAM3Torch for SAM-style interactive segmentation.

    Replaces inference.models.sam3.visual_segmentation.Sam3ForInteractiveImageSegmentation.
    Handles Sam2EmbeddingRequest / Sam2SegmentationRequest with point/box prompts via
    SAM3Torch.embed_images and SAM3Torch.segment_with_visual_prompts (sharing Sam2 request/response
    schemas, as the legacy class did).
    """

    def __init__(
        self,
        *args,
        model_id: str = "sam3/sam3_final",
        api_key: Optional[str] = None,
        low_res_logits_cache_size: int = SAM3_MAX_LOGITS_CACHE_SIZE,
        embedding_cache_size: int = SAM3_MAX_EMBEDDING_CACHE_SIZE,
        **kwargs,
    ):
        super().__init__()
        self.metrics = {"num_inferences": 0, "avg_inference_time": 0.0}
        self.api_key = api_key if api_key else API_KEY
        self.task_type = "unsupervised-segmentation"

        sam3_image_embeddings_cache = Sam3ImageEmbeddingsInMemoryCache.init(
            size_limit=embedding_cache_size,
            send_to_cpu=True,
        )
        sam3_low_resolution_masks_cache = Sam3LowResolutionMasksInMemoryCache.init(
            size_limit=low_res_logits_cache_size,
            send_to_cpu=True,
        )
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
            sam3_image_embeddings_cache=sam3_image_embeddings_cache,
            sam3_low_resolution_masks_cache=sam3_low_resolution_masks_cache,
            sam3_allow_client_generated_hash_ids=True,
            weights_provider_extra_headers=extra_weights_provider_headers,
            backend=backend,
            **kwargs,
        )

    @usage_collector("model")
    def infer_from_request(self, request: Sam2InferenceRequest):
        t1 = perf_counter()
        if isinstance(request, Sam2EmbeddingRequest):
            _, _, image_id = self.embed_image(**request.dict())
            return Sam2EmbeddingResponse(time=perf_counter() - t1, image_id=image_id)
        if isinstance(request, Sam2SegmentationRequest):
            masks, scores, low_res_logits = self.segment_image(**request.dict())
            if request.format == "json" or request.format == "polygon":
                return _build_polygon_response(
                    masks=masks,
                    scores=scores,
                    inference_start_timestamp=t1,
                )
            if request.format == "rle":
                return _build_rle_response(
                    masks=masks,
                    scores=scores,
                    inference_start_timestamp=t1,
                )
            if request.format == "binary":
                buf = BytesIO()
                np.savez_compressed(buf, masks=masks, low_res_masks=low_res_logits)
                buf.seek(0)
                return buf.getvalue()
            raise ValueError(f"Invalid format {request.format}")
        raise ValueError(f"Invalid request type {type(request)}")

    def preproc_image(self, image: InferenceRequestImage):
        if image is not None:
            return load_image_rgb(image)
        return None

    def embed_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        **kwargs,
    ):
        loaded_image = self.preproc_image(image)
        if loaded_image is None:
            raise ValueError("Image must be provided to handle this request.")
        embeddings = self._model.embed_images(
            images=loaded_image, image_hashes=image_id, **kwargs
        )[0]
        # The interactive backend stores opaque processor state, not array embeddings.
        # Preserve the legacy public shape: dict of "image_embed" / "high_res_feats".
        embedding_dict = {
            "image_embed": None,
            "high_res_feats": None,
            "state": embeddings.embeddings,
        }
        return embedding_dict, embeddings.image_size_hw, embeddings.image_hash

    def segment_image(
        self,
        image: Optional[InferenceRequestImage],
        image_id: Optional[str] = None,
        prompts: Optional[Union[Sam2PromptSet, dict]] = None,
        multimask_output: Optional[bool] = True,
        mask_input: Optional[Union[np.ndarray, List[List[List[float]]]]] = None,
        save_logits_to_cache: bool = False,
        load_logits_from_cache: bool = False,
        **kwargs,
    ):
        load_logits_from_cache = (
            load_logits_from_cache and not DISABLE_SAM3_LOGITS_CACHE
        )
        save_logits_to_cache = save_logits_to_cache and not DISABLE_SAM3_LOGITS_CACHE
        loaded_image = self.preproc_image(image)

        if prompts is not None:
            if isinstance(prompts, dict):
                prompts = Sam2PromptSet(**prompts)
        else:
            prompts = Sam2PromptSet()
        args = prompts.to_sam2_inputs()
        args = _pad_points(args)
        if not any(args.values()):
            args = {"point_coords": [[0, 0]], "point_labels": [-1], "box": None}
        if args["point_coords"] is not None:
            args["point_coords"] = np.array(args["point_coords"])
        if args["point_labels"] is not None:
            args["point_labels"] = np.array(args["point_labels"])
        if args["box"] is not None:
            args["box"] = np.array(args["box"])
        if mask_input is not None and isinstance(mask_input, list):
            mask_input = np.array(mask_input)

        prediction = self._model.segment_with_visual_prompts(
            images=loaded_image,
            image_hashes=image_id,
            point_coordinates=args["point_coords"],
            point_labels=args["point_labels"],
            boxes=args["box"],
            mask_input=mask_input,
            multi_mask_output=multimask_output,
            return_logits=True,
            load_from_mask_input_cache=load_logits_from_cache,
            save_to_mask_input_cache=save_logits_to_cache,
            use_embeddings_cache=True,
        )[0]
        return _choose_most_confident_sam_prediction(
            masks=prediction.masks.cpu().numpy(),
            scores=prediction.scores.cpu().numpy(),
            low_resolution_logits=prediction.logits.cpu().numpy(),
        )


def _pad_points(args: Dict[str, Any]) -> Dict[str, Any]:
    args = copy.deepcopy(args)
    if args["point_coords"] is not None:
        max_len = max(max(len(prompt) for prompt in args["point_coords"]), 1)
        for prompt in args["point_coords"]:
            for _ in range(max_len - len(prompt)):
                prompt.append([0, 0])
        for label in args["point_labels"]:
            for _ in range(max_len - len(label)):
                label.append(-1)
    else:
        if args["point_labels"] is not None:
            raise ValueError(
                "Can't have point labels without corresponding point coordinates"
            )
    return args


def _choose_most_confident_sam_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    low_resolution_logits: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=0)
        scores = np.expand_dims(scores, axis=0)
        low_resolution_logits = np.expand_dims(low_resolution_logits, axis=0)
    selected_masks, selected_scores, selected_logits = [], [], []
    for mask, score, low_res in zip(masks, scores, low_resolution_logits):
        max_idx = int(np.argsort(score)[-1])
        selected_masks.append(mask[max_idx])
        selected_scores.append(score[max_idx].item())
        selected_logits.append(low_res[max_idx])
    return (
        np.asarray(selected_masks),
        np.asarray(selected_scores),
        np.asarray(selected_logits),
    )


def _build_polygon_response(
    masks: np.ndarray,
    scores: np.ndarray,
    inference_start_timestamp: float,
) -> Sam2SegmentationResponse:
    predictions: List[Sam2SegmentationPrediction] = []
    polygons = masks2multipoly(masks >= MASK_THRESHOLD)
    for poly, score in zip(polygons, scores):
        predictions.append(
            Sam2SegmentationPrediction(
                masks=[m.tolist() for m in poly],
                confidence=float(score),
                format="polygon",
            )
        )
    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )


def _build_rle_response(
    masks: np.ndarray,
    scores: np.ndarray,
    inference_start_timestamp: float,
) -> Sam2SegmentationResponse:
    predictions: List[Sam2SegmentationPrediction] = []
    for mask, score in zip(masks, scores):
        mb = (mask >= MASK_THRESHOLD).astype(np.uint8)
        rle = mask_utils.encode(np.asfortranarray(mb))
        rle["counts"] = rle["counts"].decode("utf-8")
        predictions.append(
            Sam2SegmentationPrediction(masks=rle, confidence=float(score), format="rle")
        )
    return Sam2SegmentationResponse(
        time=perf_counter() - inference_start_timestamp,
        predictions=predictions,
    )
